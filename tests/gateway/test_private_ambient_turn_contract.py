"""Minimal contract for private ambient gateway turns."""

import asyncio

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource


class _PrivateQueueAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> dict:
        return {}

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True)


class _TypingLifecycleAdapter(_PrivateQueueAdapter):
    """Production-shaped Telegram adapter with observable typing transport."""

    def __init__(self) -> None:
        super().__init__(PlatformConfig(enabled=True, token="test-token"), Platform.TELEGRAM)
        self.sent: list[str] = []
        self.typing_calls: list[str] = []
        self.stop_calls: list[str] = []

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        self.typing_calls.append(chat_id)

    async def stop_typing(self, chat_id: str) -> None:
        self.stop_calls.append(chat_id)


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="private-ambient-chat",
        chat_type="dm",
        user_id="private-ambient-user",
    )


@pytest.mark.asyncio
async def test_private_event_is_accepted_into_a_separate_pending_queue() -> None:
    adapter = object.__new__(_PrivateQueueAdapter)
    adapter._pending_private_messages = {}
    adapter._pending_ingress_order = 0
    event = MessageEvent(
        text="private wake",
        source=_source(),
        internal=True,
        allow_gateway_control=False,
        metadata={"hermes_private_turn": True},
    )

    await adapter._handle_message_while_active(event, "session-key")

    assert event._gateway_accepted is True
    assert adapter._pending_private_messages == {"session-key": [event]}


@pytest.mark.asyncio
async def test_queued_private_turn_types_only_when_running_then_stops_after_no_reply_release() -> None:
    """A queued heartbeat owns a typing lifecycle only after its turn starts, even when it is silent."""
    adapter = _TypingLifecycleAdapter()
    session_key = "agent:main:telegram:dm:private-ambient-chat"
    ordinary_started = asyncio.Event()
    release_ordinary = asyncio.Event()
    private_started = asyncio.Event()
    release_private = asyncio.Event()
    transformed = []

    async def handler(event):
        if (event.metadata or {}).get("hermes_private_turn"):
            private_started.set()
            await release_private.wait()
            return "private raw completion"
        ordinary_started.set()
        await release_ordinary.wait()
        return "ordinary response"

    adapter._message_handler = handler
    ordinary = MessageEvent(text="ordinary", source=_source())
    private = MessageEvent(
        text="private heartbeat", source=_source(), internal=True,
        metadata={"hermes_private_turn": True},
    )
    setattr(
        private,
        "_injected_response_transform",
        lambda response, key: transformed.append((response, key)) or "NO_REPLY",
    )

    await adapter.handle_message(ordinary)
    await asyncio.wait_for(ordinary_started.wait(), timeout=1.0)
    await asyncio.wait_for(
        _wait_until(lambda: len(adapter.typing_calls) == 1), timeout=1.0,
    )

    await adapter.handle_message(private)
    assert adapter._pending_private_messages[session_key] == [private]
    await asyncio.sleep(0.05)
    assert len(adapter.typing_calls) == 1  # Still only the unrelated ordinary turn.

    release_ordinary.set()
    await asyncio.wait_for(private_started.wait(), timeout=1.0)
    await asyncio.wait_for(
        _wait_until(lambda: len(adapter.typing_calls) == 2), timeout=1.0,
    )

    release_private.set()
    await asyncio.wait_for(
        _wait_until(lambda: session_key not in adapter._session_tasks), timeout=1.0,
    )
    calls_after_finish = len(adapter.typing_calls)
    await asyncio.sleep(0.05)

    assert transformed == [("private raw completion", session_key)]
    assert adapter.sent == ["ordinary response"]
    assert len(adapter.stop_calls) >= 2
    assert len(adapter.typing_calls) == calls_after_finish
    assert session_key not in adapter._active_sessions


async def _wait_until(predicate) -> None:
    while not predicate():
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_private_transform_failure_stops_typing_without_delivering_raw_output() -> None:
    """A malformed/rejected private completion cannot leave a Telegram refresh owner behind."""
    adapter = _TypingLifecycleAdapter()
    session_key = "agent:main:telegram:dm:private-ambient-chat"
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(_event):
        started.set()
        await release.wait()
        return "raw private completion"

    def reject(_response, _key):
        raise ValueError("completion rejected")

    event = MessageEvent(
        text="private heartbeat", source=_source(), internal=True,
        metadata={"hermes_private_turn": True},
    )
    setattr(event, "_injected_response_transform", reject)
    adapter._message_handler = handler

    await adapter.handle_message(event)
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.wait_for(_wait_until(lambda: adapter.typing_calls), timeout=1.0)
    release.set()
    await asyncio.wait_for(
        _wait_until(lambda: session_key not in adapter._session_tasks), timeout=1.0,
    )

    assert adapter.sent == []
    assert adapter.stop_calls
    assert session_key not in adapter._active_sessions


@pytest.mark.asyncio
async def test_private_notification_release_stops_typing_after_delivery() -> None:
    """A released heartbeat notification ends its refresh owner after the visible send."""
    adapter = _TypingLifecycleAdapter()
    session_key = "agent:main:telegram:dm:private-ambient-chat"
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(_event):
        started.set()
        await release.wait()
        return "raw private completion"

    event = MessageEvent(
        text="private heartbeat", source=_source(), internal=True,
        metadata={"hermes_private_turn": True},
    )
    setattr(event, "_injected_response_transform", lambda _response, _key: "heartbeat notification")
    adapter._message_handler = handler

    await adapter.handle_message(event)
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.wait_for(_wait_until(lambda: adapter.typing_calls), timeout=1.0)
    release.set()
    await asyncio.wait_for(
        _wait_until(lambda: session_key not in adapter._session_tasks), timeout=1.0,
    )

    assert adapter.sent == ["heartbeat notification"]
    assert adapter.stop_calls
    assert session_key not in adapter._active_sessions


@pytest.mark.asyncio
async def test_cancelled_private_turn_stops_typing_and_releases_its_session_guard() -> None:
    """Cancellation is terminal for the private turn's typing owner and session guard."""
    adapter = _TypingLifecycleAdapter()
    session_key = "agent:main:telegram:dm:private-ambient-chat"
    started = asyncio.Event()

    async def handler(_event):
        started.set()
        await asyncio.Event().wait()

    event = MessageEvent(
        text="private heartbeat", source=_source(), internal=True,
        metadata={"hermes_private_turn": True},
    )
    adapter._message_handler = handler

    await adapter.handle_message(event)
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.wait_for(_wait_until(lambda: adapter.typing_calls), timeout=1.0)
    owner = adapter._session_tasks[session_key]
    owner.cancel()
    await asyncio.gather(owner, return_exceptions=True)
    await asyncio.wait_for(
        _wait_until(lambda: session_key not in adapter._session_tasks), timeout=1.0,
    )

    assert adapter.stop_calls
    assert session_key not in adapter._active_sessions


@pytest.mark.asyncio
async def test_ordinary_turn_completion_stops_typing_and_releases_its_session_guard() -> None:
    """The same task-owned cleanup covers ordinary Telegram turns."""
    adapter = _TypingLifecycleAdapter()
    session_key = "agent:main:telegram:dm:private-ambient-chat"
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(_event):
        started.set()
        await release.wait()
        return "ordinary response"

    adapter._message_handler = handler
    await adapter.handle_message(MessageEvent(text="ordinary", source=_source()))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.wait_for(_wait_until(lambda: adapter.typing_calls), timeout=1.0)
    release.set()
    await asyncio.wait_for(
        _wait_until(lambda: session_key not in adapter._session_tasks), timeout=1.0,
    )

    assert adapter.sent == ["ordinary response"]
    assert adapter.stop_calls
    assert session_key not in adapter._active_sessions


def test_private_and_public_pending_events_are_not_merged() -> None:
    adapter = object.__new__(_PrivateQueueAdapter)
    adapter._pending_ingress_order = 2
    private = MessageEvent(text="private wake", source=_source(), internal=True)
    public = MessageEvent(text="public follow-up", source=_source())
    setattr(private, "_hermes_pending_order", 1)
    setattr(public, "_hermes_pending_order", 2)
    adapter._pending_private_messages = {"session-key": [private]}
    adapter._pending_messages = {"session-key": public}

    assert adapter.get_pending_message("session-key") is private
    assert adapter.get_pending_message("session-key") is public


def test_private_turn_skips_normal_session_db_and_external_memory() -> None:
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    memory_manager = MagicMock()
    setattr(agent, "_gateway_private_turn", True)
    setattr(agent, "_persist_disabled", False)
    setattr(agent, "_session_persist_lock", None)
    setattr(agent, "_session_db", MagicMock())
    setattr(agent, "_memory_manager", memory_manager)
    setattr(agent, "session_id", "private-session")

    assert agent._flush_messages_to_session_db_unlocked(
        [{"role": "user", "content": "private input"}], []
    ) is None
    agent._session_db.append_messages_batch.assert_not_called()  # type: ignore[attr-defined]

    agent._sync_external_memory_for_turn(
        original_user_message="private input",
        final_response="private output",
        interrupted=False,
    )
    memory_manager.sync_all.assert_not_called()
    memory_manager.queue_prefetch_all.assert_not_called()


def test_private_output_is_replaced_before_gateway_delivery(monkeypatch) -> None:
    from agent.turn_finalizer import _apply_output_hooks

    observed = []
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kwargs: observed.append((name, kwargs)) or [],
    )
    agent = SimpleNamespace(
        _gateway_private_turn=True,
        _persist_disabled=False,
        session_id="private-session",
        model="test-model",
    )

    response, transformed, _ = _apply_output_hooks(
        agent,
        "private model output",
        MagicMock(),
        platform="telegram",
        effective_task_id="private-task",
        turn_id="private-turn",
        original_user_message="private input",
        messages=[],
    )

    assert response == "private model output"
    assert transformed is False
    assert observed == []


def test_private_response_requires_its_process_local_release_callback() -> None:
    event = MessageEvent(
        text="private wake",
        source=_source(),
        internal=True,
        metadata={"hermes_private_turn": True},
    )

    assert BasePlatformAdapter._release_private_response(
        event, "raw private output", "session-key"
    ) is None


def test_private_no_reply_release_is_suppressed() -> None:
    event = MessageEvent(
        text="private wake",
        source=_source(),
        internal=True,
        metadata={"hermes_private_turn": True},
    )
    setattr(event, "_injected_response_transform", lambda _response, _session_key: "NO_REPLY")

    assert BasePlatformAdapter._release_private_response(
        event, "raw private output", "session-key"
    ) is None


def test_private_response_uses_only_its_process_local_release_callback() -> None:
    event = MessageEvent(
        text="private wake",
        source=_source(),
        internal=True,
        metadata={"hermes_private_turn": True},
    )
    observed = []
    event._injected_response_transform = lambda response_text, session_key: (
        observed.append((response_text, session_key)) or "released notification"
    )

    assert BasePlatformAdapter._release_private_response(
        event, "raw private output", "session-key"
    ) == "released notification"
    assert BasePlatformAdapter._release_private_response(
        event, "second raw output", "session-key"
    ) is None
    assert observed == [("raw private output", "session-key")]


def test_injected_turn_lifecycle_callback_is_terminal_once() -> None:
    event = MessageEvent(text="private wake", source=_source(), internal=True)
    observed = []
    event._injected_turn_state_callback = observed.append

    BasePlatformAdapter._report_injected_turn_state(event, "started")
    BasePlatformAdapter._report_injected_turn_state(event, "finished")
    BasePlatformAdapter._report_injected_turn_state(event, "cancelled")

    assert observed == ["started", "finished"]


def test_ordinary_plugin_steer_is_an_explicit_opt_in() -> None:
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)
    context = PluginContext(
        PluginManifest(name="ordinary-plugin", key="ordinary-plugin", source="user"), manager,
    )
    context._gateway_injection_allowed = lambda: True

    assert context.inject_message("wake", session_key="session-key", busy_policy="steer") is True
    assert injector.call_args.kwargs["busy_policy"] == "steer"
