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
    assert messages == [history, private_user, private_tool]