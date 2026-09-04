from types import SimpleNamespace
from typing import Any

from agent.turn_finalizer import finalize_turn


class FakeAgent:
    def __init__(self):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = True
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages: list[dict[str, Any]] | None = None
        self._persist_user_message_idx: int | None = None
        self._persist_user_message_override: Any = None
        self._persist_user_message_timestamp: float | None = None
        self._gateway_private_turn = False

    def _handle_max_iterations(self, messages, api_call_count):
        raise AssertionError("not expected")

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        # Capture the durable write before finalization restores API-local
        # guidance to the returned/live transcript.
        self.persisted_messages = [dict(message) for message in messages]

    def _apply_persist_user_message_override(self, messages):
        idx = self._persist_user_message_idx
        override = self._persist_user_message_override
        if idx is not None and override is not None:
            messages[idx]["content"] = override

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def test_private_turn_never_notifies_post_llm_observers(monkeypatch):
    """The observer boundary must not receive private raw turn material.

    This invokes the real finalizer rather than only assigning a flag around
    ``_persist_session``. The former presentation test missed this observer
    path, which receives the full raw conversation history on a live turn.
    """
    observed = []

    def capture_hook(name, **kwargs):
        if name == "post_llm_call":
            observed.append(kwargs)
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", capture_hook)
    agent = FakeAgent()
    agent._gateway_private_turn = True
    messages = [
        {"role": "user", "content": "PRIVATE injected envelope"},
        {"role": "assistant", "content": "PRIVATE raw assistant output"},
    ]

    result = finalize_turn(
        agent,
        final_response="PRIVATE raw assistant output",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="private-task",
        turn_id="private-turn",
        user_message="PRIVATE injected envelope",
        original_user_message="PRIVATE injected envelope",
        _should_review_memory=False,
        _turn_exit_reason="completed",
    )

    assert result["final_response"] == "NO_REPLY"
    assert observed == []


def test_normal_turn_still_notifies_post_llm_observers(monkeypatch):
    observed = []
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kwargs: observed.append((name, kwargs)) or [],
    )
    agent = FakeAgent()
    agent._gateway_private_turn = False

    finalize_turn(
        agent,
        final_response="ordinary answer",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "ordinary question"}],
        conversation_history=[],
        effective_task_id="normal-task",
        turn_id="normal-turn",
        user_message="ordinary question",
        original_user_message="ordinary question",
        _should_review_memory=False,
        _turn_exit_reason="completed",
    )

    post_calls = [kwargs for name, kwargs in observed if name == "post_llm_call"]
    assert len(post_calls) == 1
    assert post_calls[0]["assistant_response"] == "ordinary answer"


def test_private_turn_never_notifies_context_engine_observer(monkeypatch):
    observed = []
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setattr(
        "agent.conversation_loop._notify_context_engine_turn_complete",
        lambda _agent, messages, **_kwargs: observed.append(messages),
    )
    agent = FakeAgent()
    agent._gateway_private_turn = True
    messages = [
        {"role": "user", "content": "PRIVATE injected envelope"},
        {"role": "assistant", "content": "PRIVATE raw assistant output"},
    ]

    finalize_turn(
        agent,
        final_response="PRIVATE raw assistant output",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="private-task",
        turn_id="private-turn",
        user_message="PRIVATE injected envelope",
        original_user_message="PRIVATE injected envelope",
        _should_review_memory=False,
        _turn_exit_reason="completed",
    )

    assert observed == []


def test_normal_turn_still_notifies_context_engine_observer(monkeypatch):
    observed = []
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setattr(
        "agent.conversation_loop._notify_context_engine_turn_complete",
        lambda _agent, messages, **_kwargs: observed.append(messages),
    )
    agent = FakeAgent()
    messages = [{"role": "user", "content": "ordinary question"}]

    finalize_turn(
        agent,
        final_response="ordinary answer",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="normal-task",
        turn_id="normal-turn",
        user_message="ordinary question",
        original_user_message="ordinary question",
        _should_review_memory=False,
        _turn_exit_reason="completed",
    )

    assert observed == [messages]


def test_private_finalization_cannot_leak_into_next_cached_normal_persist(monkeypatch):
    """A private turn must leave only pre-turn history in the agent cache.

    A cached gateway agent starts the next normal turn from ``_session_messages``.
    If finalization restores the private live transcript after ``_persist_session``
    strips it, that later normal persist serializes the prior private prompt and
    response despite the private turn itself having skipped persistence.
    """
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *_a, **_kw: [])

    class CachedHistoryAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self.durable_writes = []

        def _persist_session(self, messages, conversation_history):
            if self._gateway_private_turn:
                self._session_messages = messages[:self._persist_user_message_idx]
            else:
                self.durable_writes.append([dict(message) for message in messages])
                self._session_messages = messages

    agent = CachedHistoryAgent()
    agent._gateway_private_turn = True
    agent._persist_user_message_idx = 1
    private_messages = [
        {"role": "assistant", "content": "ordinary cached history"},
        {"role": "user", "content": "PRIVATE injected envelope"},
        {"role": "assistant", "content": "PRIVATE raw assistant output"},
    ]

    finalize_turn(
        agent,
        final_response="PRIVATE raw assistant output",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=private_messages,
        conversation_history=private_messages[:1],
        effective_task_id="private-task",
        turn_id="private-turn",
        user_message="PRIVATE injected envelope",
        original_user_message="PRIVATE injected envelope",
        _should_review_memory=False,
        _turn_exit_reason="completed",
    )

    agent._gateway_private_turn = False
    agent._persist_session(
        agent._session_messages + [{"role": "user", "content": "ordinary follow-up"}],
        [],
    )

    persisted_text = "\n".join(
        str(message.get("content", ""))
        for message in agent.durable_writes[-1]
    )
    assert "PRIVATE injected envelope" not in persisted_text
    assert "PRIVATE raw assistant output" not in persisted_text
    assert "ordinary cached history" in persisted_text
    assert "ordinary follow-up" in persisted_text


def test_private_turn_never_micro_compacts_current_transcript(monkeypatch):
    """The post-turn compactor must not archive private prompt/output rows."""
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *_a, **_kw: [])

    class RecordingCompressor:
        _micro_compact_enabled = True

        def __init__(self):
            self.canonical_writes = []

        def _micro_compact(self, messages):
            self.canonical_writes.append([message.get("content") for message in messages])
            return list(messages)

    agent = FakeAgent()
    agent._gateway_private_turn = True
    compressor = RecordingCompressor()
    agent.context_compressor = compressor
    messages = [
        {"role": "user", "content": "PRIVATE current prompt"},
        {"role": "assistant", "content": "PRIVATE current output"},
    ]

    finalize_turn(
        agent,
        final_response="PRIVATE current output",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="private-task",
        turn_id="private-turn",
        user_message="PRIVATE current prompt",
        original_user_message="PRIVATE current prompt",
        _should_review_memory=False,
        _turn_exit_reason="completed",
    )

    assert compressor.canonical_writes == []


def test_private_turn_never_saves_trajectory(monkeypatch):
    saved = []
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    agent._gateway_private_turn = True
    agent._save_trajectory = lambda *args: saved.append(args)

    finalize_turn(
        agent,
        final_response="PRIVATE raw assistant output",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[
            {"role": "user", "content": "PRIVATE injected envelope"},
            {"role": "assistant", "content": "PRIVATE raw assistant output"},
        ],
        conversation_history=[],
        effective_task_id="private-task",
        turn_id="private-turn",
        user_message="PRIVATE injected envelope",
        original_user_message="PRIVATE injected envelope",
        _should_review_memory=False,
        _turn_exit_reason="completed",
    )

    assert saved == []


def test_final_response_closes_tool_tail_before_persistence(monkeypatch):
    """A recovered/previewed final response must be durable in session history.

    Regression for turns where the caller receives a non-empty final_response,
    but the message transcript still ends at a tool result. If persisted that
    way, the next turn reloads a stale/malformed history and can appear to loop
    because the assistant's visible final answer is missing from durable state.
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "I'll check.",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "terminal", "content": "ok"},
    ]

    result = finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="do it",
        original_user_message="do it",
        _should_review_memory=False,
        _turn_exit_reason="fallback_prior_turn_content",
    )

    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"] == "Done."
    assert isinstance(result["messages"][-1]["timestamp"], float)
    assert agent.persisted_messages is not None
    assert agent.persisted_messages[-1] == result["messages"][-1]


def test_fallback_timestamp_survives_delayed_sqlite_persistence(
    monkeypatch, tmp_path
):
    """The durable row records message creation, not the later DB flush."""
    from hermes_state import SessionDB

    created_at = 1_781_976_577.25
    persisted_at = created_at + 600
    monkeypatch.setattr("agent.message_metadata.wall_time", lambda: created_at)
    monkeypatch.setattr("hermes_state.time.time", lambda: persisted_at)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess-test", source="cli")
    agent = FakeAgent()

    def persist_to_sqlite(messages, _conversation_history):
        db.replace_messages(agent.session_id, messages)
        agent.persisted_messages = db.get_messages_as_conversation(agent.session_id)

    agent._persist_session = persist_to_sqlite
    messages = [
        {"role": "user", "content": "do it", "timestamp": created_at - 1},
        {"role": "tool", "content": "ok", "tool_call_id": "call-1"},
    ]

    finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="do it",
        original_user_message="do it",
        _should_review_memory=False,
        _turn_exit_reason="fallback_prior_turn_content",
    )

    assert agent.persisted_messages[-1]["timestamp"] == created_at
    assert agent.persisted_messages[-1]["timestamp"] != persisted_at


def test_final_response_fills_pure_tool_call_tail(monkeypatch):
    """A tail assistant row that is a *pure tool-call turn* carries no answer.

    The role check alone ("tail is assistant ⇒ nothing to do") leaves the
    #43849/#44100 invariant unmet when the tail is ``assistant(tool_calls)``
    with no text of its own: the caller and the gateway already delivered
    ``final_response``, but it never reaches the transcript. The next turn then
    replays the user backlog and the model re-answers it — the exact symptom
    that block exists to prevent.
    """
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}
            ],
        },
    ]

    result = finalize_turn(
        agent,
        final_response="Here is your answer.",
        api_call_count=3,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="t",
        turn_id="tid",
        user_message="q",
        original_user_message="q",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    persisted = agent.persisted_messages
    assert any(
        m.get("role") == "assistant" and m.get("content") == result["final_response"]
        for m in persisted
    ), "delivered final_response never reached the durable transcript"
    # Filled in place — no assistant→assistant pair, tool_calls preserved.
    assert persisted[-1]["content"] == "Here is your answer."
    assert persisted[-1]["tool_calls"]
    assert sum(1 for m in persisted if m.get("role") == "assistant") == 1






def test_final_response_fill_invalidates_flush_scan_cursor():
    """The fill's marker pop must invalidate the bounded flush-scan cursor.

    The cursor (run_agent.py) skips the identity-matched prefix of its
    previous snapshot assuming no live dict loses ``_db_persisted`` in place
    — the fill is the one path that pops it. Without invalidation, the
    turn-end flush skips the filled row as 'already stamped' and the
    delivered answer never reaches state.db (the #43849 class resurfacing).
    """
    agent = FakeAgent()
    agent._db_flush_scan_prefix = ["prior-snapshot"]
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}
            ],
            "_db_persisted": True,
        },
    ]

    finalize_turn(
        agent,
        final_response="Here is your answer.",
        api_call_count=3,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="t",
        turn_id="tid",
        user_message="q",
        original_user_message="q",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    assert agent._db_flush_scan_prefix is None


def test_empty_final_response_recovers_stream_buffer_into_blank_assistant_row(
    monkeypatch, tmp_path
):
    """#95514: empty terminal completion must not persist content='' over a live stream.

    After a tool result the incremental flush can leave a blank assistant tail.
    If finalize_turn then receives final_response='' while the stream buffer
    still holds the delivered text, that blank row is the durable transcript —
    Desktop reload shows nothing even though the user already saw the answer.
    """
    from hermes_state import SessionDB

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess-test", source="cli")
    agent = FakeAgent()
    agent._current_streamed_assistant_text = "Already streamed to the user."

    def persist_to_sqlite(messages, _conversation_history):
        db.replace_messages(agent.session_id, messages)
        agent.persisted_messages = db.get_messages_as_conversation(agent.session_id)

    agent._persist_session = persist_to_sqlite
    messages = [
        {"role": "user", "content": "summarize"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "name": "terminal", "content": "ok"},
        {"role": "assistant", "content": "", "_db_persisted": True},
    ]

    result = finalize_turn(
        agent,
        final_response="",
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="summarize",
        original_user_message="summarize",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    assert result["final_response"] == "Already streamed to the user."
    assistants = [m for m in agent.persisted_messages if m.get("role") == "assistant"]
    assert assistants[-1]["content"] == "Already streamed to the user."
    assert assistants[0].get("tool_calls")
    assert sum(1 for m in assistants) == 2

    reopened = SessionDB(db_path=tmp_path / "state.db")
    reloaded = reopened.get_messages_as_conversation("sess-test")
    assert reloaded[-1]["role"] == "assistant"
    assert reloaded[-1]["content"] == "Already streamed to the user."
    assert sum(1 for m in reloaded if m.get("role") == "assistant") == 2


def test_failed_turn_does_not_recover_stream_buffer_as_final_response(monkeypatch):
    """A failed turn must not invent a successful answer from the stream buffer."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    agent._current_streamed_assistant_text = "partial tokens"

    result = finalize_turn(
        agent,
        final_response="",
        api_call_count=1,
        interrupted=False,
        failed=True,
        messages=[{"role": "user", "content": "q"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="q",
        original_user_message="q",
        _should_review_memory=False,
        _turn_exit_reason="error",
    )

    assert result["final_response"] in ("", None)
    assert result["failed"] is True


def test_delivery_only_reasoning_excerpt_does_not_fill_blank_assistant(monkeypatch):
    """Labeled empty-terminal excerpt is delivery-only, not durable content.

    ``empty_response_exhausted`` sets final_response to a labeled reasoning
    excerpt while the transcript tail is a blank/sentinel assistant without
    tool_calls. Filling that tail would persist the excerpt and break replay
    (test_empty_terminal_reasoning_surface).
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    excerpt = (
        "⚠️ The model produced only internal reasoning and no final answer, "
        "despite retries. Its last reasoning, which may contain the answer:\n\n"
        "The answer is 42"
    )
    messages = [
        {"role": "user", "content": "what is the answer?"},
        {"role": "assistant", "content": ""},
    ]

    result = finalize_turn(
        agent,
        final_response=excerpt,
        api_call_count=6,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="what is the answer?",
        original_user_message="what is the answer?",
        _should_review_memory=False,
        _turn_exit_reason="empty_response_exhausted",
    )

    assert result["final_response"] == excerpt
    assert not any(
        m.get("role") == "assistant"
        and "only internal reasoning" in (m.get("content") or "")
        for m in result["messages"]
    )

