"""Private gateway turns preserve provenance for automatic recall exclusion."""

from unittest.mock import MagicMock


def test_private_turn_marks_only_its_user_assistant_and_tool_messages():
    from run_agent import _mark_private_turn_messages

    history = {"role": "assistant", "content": "ordinary history"}
    private_user = {"role": "user", "content": "private wake"}
    private_assistant = {"role": "assistant", "content": "private result"}
    private_tool = {"role": "tool", "content": "private tool result"}
    messages = [history, private_user, private_assistant, private_tool]

    _mark_private_turn_messages(messages, start_index=1)

    assert "hermes_private_turn" not in history
    assert all(message["hermes_private_turn"] is True for message in messages[1:])


def test_private_turn_persistence_discards_private_tail_from_cache_log_and_db(monkeypatch):
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._gateway_private_turn = True
    agent._persist_user_message_idx = 1
    agent._session_persist_lock = None
    agent._session_db = None
    agent._drop_trailing_empty_response_scaffolding = MagicMock()
    agent._save_session_log = MagicMock()
    agent._flush_messages_to_session_db = MagicMock()
    monkeypatch.setattr("agent.agent_runtime_helpers.note_turn_persisted", lambda _agent: None)

    history = {"role": "assistant", "content": "ordinary history"}
    private_user = {"role": "user", "content": "private wake"}
    private_tool = {"role": "tool", "content": "PRIVATE RAW TOOL OUTPUT"}
    messages = [history, private_user, private_tool]

    agent._persist_session(messages)

    assert agent._session_messages == [history]
    agent._save_session_log.assert_not_called()
    agent._flush_messages_to_session_db.assert_not_called()
    assert "hermes_private_turn" not in history
    assert private_user["hermes_private_turn"] is True
    assert private_tool["hermes_private_turn"] is True


def test_private_turn_direct_flush_marks_tail_before_skipping_db():
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._gateway_private_turn = True
    agent._persist_user_message_idx = 1
    agent._session_persist_lock = None
    agent._flush_messages_to_session_db_unlocked = MagicMock()

    history = {"role": "assistant", "content": "ordinary history"}
    private_user = {"role": "user", "content": "private wake"}
    private_tool = {"role": "tool", "content": "PRIVATE RAW TOOL OUTPUT"}
    messages = [history, private_user, private_tool]

    assert agent._flush_messages_to_session_db(messages, [history]) is None

    agent._flush_messages_to_session_db_unlocked.assert_not_called()
    assert "hermes_private_turn" not in history
    assert private_user["hermes_private_turn"] is True
    assert private_tool["hermes_private_turn"] is True


def test_later_public_flush_cannot_persist_a_stamped_private_tail(tmp_path):
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._gateway_private_turn = False
    agent._session_persist_lock = None
    agent._session_db = MagicMock()
    agent._session_db.db_path = str(tmp_path / "state.db")
    agent._session_db_created = True
    agent.session_id = "session-1"
    agent._last_flushed_db_idx = 0
    agent._persist_disabled = False
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._pending_cli_user_message = None
    agent._persist_user_message_timestamp = None
    agent._persist_user_message_idx = 1
    agent._persist_user_message_override = None

    private_user = {
        "role": "user",
        "content": "private wake",
        "hermes_private_turn": True,
    }
    public_user = {"role": "user", "content": "ordinary follow-up"}

    agent._flush_messages_to_session_db([private_user, public_user], [])

    rows = agent._session_db.append_messages_batch.call_args.kwargs["messages"]
    assert [row["content"] for row in rows] == ["ordinary follow-up"]
