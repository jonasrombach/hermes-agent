"""Tests for plugin-triggered turns in existing gateway sessions."""

import asyncio
import concurrent.futures
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    PlatformConfig,
    TextDebounceState,
)
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
from gateway.session_state import TurnState
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _entry(*, origin=True) -> SessionEntry:
    source = None
    if origin:
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="42",
            chat_type="dm",
            user_id="42",
            user_name="tester",
        )
    now = datetime.now()
    return SessionEntry(
        session_key="agent:main:telegram:dm:42",
        session_id="session-42",
        created_at=now,
        updated_at=now,
        origin=source,
        platform=Platform.TELEGRAM,
    )


def _runner(entry: SessionEntry | None, adapter=None) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, lookup_by_session_key=AsyncMock(return_value=entry)
    )
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
    runner._profile_adapters = {}
    runner._running = True
    runner._draining = False
    runner._background_tasks = set()
    runner._is_user_authorized = MagicMock(return_value=True)
    return runner


class _RoutingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("network send is not expected")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


@pytest.mark.asyncio
async def test_plugin_context_routes_through_live_gateway_to_existing_session(
    tmp_path,
    monkeypatch,
):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({
            "plugins": {"entries": {"notify-plugin": {"allow_gateway_injection": True}}}
        })
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    source = _entry().origin
    entry = store.get_or_create_session(source)
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    adapter._active_sessions[entry.session_key] = asyncio.Event()
    pending_user_event = MessageEvent(
        text="human follow-up",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["human.jpg"],
        media_types=["image/jpeg"],
    )
    adapter._pending_messages[entry.session_key] = pending_user_event

    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    runner._draining = False
    runner._background_tasks = set()
    runner._queued_events = {}
    runner._is_user_authorized = MagicMock(return_value=True)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="notify-plugin", key="notify-plugin", source="user"),
        manager,
    )

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner._install_plugin_message_injector()
        assert (
            context.inject_message(
                "/approve always",
                session_key=entry.session_key,
            )
            is True
        )
        task = next(iter(runner._background_tasks))
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

        assert adapter._pending_messages[entry.session_key] is pending_user_event
        queued = runner._queued_events[entry.session_key][0]
        assert pending_user_event.text == "human follow-up"
        assert pending_user_event.media_urls == ["human.jpg"]
        assert pending_user_event.allow_gateway_control is True
        assert queued.text == "/approve always"
        assert queued.allow_gateway_control is False
        assert queued.metadata["gateway_session_id"] == entry.session_id
        adapter._message_handler.assert_not_awaited()

        runner._clear_plugin_message_injector()
        assert manager.has_gateway_message_injector is False


@pytest.mark.asyncio
async def test_dispatch_uses_stored_origin_and_adapter_message_path():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key,
        content="check the deployment",
        plugin_id="notify-plugin",
    )

    assert accepted is True
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "check the deployment"
    assert event.internal is True
    assert event.allow_gateway_control is False
    assert event.get_command() is None
    assert event.source == entry.origin
    assert event.source is not entry.origin
    runner._is_user_authorized.assert_called_once_with(
        event.source,
        allow_adapter_delegation=False,
    )
    assert event.metadata == {
        "hermes_plugin_id": "notify-plugin",
        "hermes_plugin_injection": True,
        "gateway_session_key": entry.session_key,
        "gateway_session_id": entry.session_id,
        "gateway_session_strict": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [False, RuntimeError("rejected")])
async def test_dispatch_propagates_adapter_acceptance_or_exception(outcome):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    if isinstance(outcome, Exception):
        adapter.handle_message.side_effect = outcome
    else:
        adapter.handle_message.return_value = outcome
    runner = _runner(_entry(), adapter)

    accepted = await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [True, False])
async def test_base_handler_returns_explicit_scheduling_acceptance(accepted):
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    event = MessageEvent(text="ordinary", source=_entry().origin)
    adapter._start_session_processing = MagicMock(return_value=accepted)

    assert await adapter.handle_message(event) is accepted


@pytest.mark.asyncio
async def test_base_handler_returns_false_when_scheduling_raises():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    event = MessageEvent(text="ordinary", source=_entry().origin)
    adapter._start_session_processing = MagicMock(side_effect=RuntimeError("boom"))

    assert await adapter.handle_message(event) is False


@pytest.mark.asyncio
async def test_private_dispatch_marks_event_without_mutating_stored_origin():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key,
        content="private background turn",
        plugin_id="notify-plugin",
        private=True,
    )

    assert accepted is True
    event = adapter.handle_message.await_args.args[0]
    assert event.metadata["hermes_private_turn"] is True
    assert event.source is not entry.origin
    assert getattr(event.source, "_hermes_private_turn") is True
    assert not hasattr(entry.origin, "_hermes_private_turn")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry", "with_adapter"),
    [
        (None, True),
        (_entry(origin=False), True),
        (_entry(), False),
    ],
)
async def test_dispatch_rejects_unroutable_session(entry, with_adapter):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(entry, adapter if with_adapter else None)

    accepted = await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_dispatch_rechecks_current_authorization(raises):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(_entry(), adapter)
    if raises:
        runner._is_user_authorized.side_effect = RuntimeError("config unavailable")
    else:
        runner._is_user_authorized.return_value = False

    accepted = await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_rejects_stored_role_only_authorization(monkeypatch):
    """A stored adapter role grant must be revalidated against current core auth."""
    for key in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)

    adapter = MagicMock(spec=BasePlatformAdapter)
    adapter.handle_message = AsyncMock()
    entry = _entry()
    entry.session_key = "agent:main:discord:dm:42"
    entry.platform = Platform.DISCORD
    source = entry.origin
    assert source is not None
    source.platform = Platform.DISCORD
    source.role_authorized = True

    runner = _runner(entry)
    runner.adapters = {Platform.DISCORD: adapter}
    runner.config = GatewayConfig()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    del runner._is_user_authorized

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key,
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_stops_when_gateway_drains_during_lookup():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(_entry(), adapter)
    lookup_started = asyncio.Event()
    release_lookup = asyncio.Event()

    async def _lookup(_session_key):
        lookup_started.set()
        await release_lookup.wait()
        return _entry()

    runner._async_session_store.lookup_by_session_key = _lookup
    dispatch = asyncio.create_task(
        runner._dispatch_plugin_message_injection(
            session_key="agent:main:telegram:dm:42",
            content="wake up",
            plugin_id="notify-plugin",
        )
    )

    await lookup_started.wait()
    runner._draining = True
    release_lookup.set()

    assert await dispatch is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_rejects_changed_origin_for_same_session_id():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    original = _entry()
    moved = _entry()
    assert moved.origin is not None
    moved.origin.chat_id = "99"
    runner = _runner(original, adapter)
    object.__setattr__(
        runner,
        "_async_session_store",
        SimpleNamespace(
            _store=runner.session_store,
            lookup_by_session_key=AsyncMock(side_effect=[original, moved]),
        ),
    )

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=original.session_key,
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_queues_non_control_plugin_text_for_exact_session():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    source = _entry().origin
    session_key = build_session_key(source)
    adapter._active_sessions[session_key] = asyncio.Event()
    event = MessageEvent(
        text="/approve always",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        allow_gateway_control=False,
        metadata={"gateway_session_key": session_key},
    )

    await adapter.handle_message(event)

    adapter._message_handler.assert_not_awaited()
    assert adapter._pending_messages[session_key] is event
    assert adapter._active_sessions[session_key].is_set() is False


@pytest.mark.asyncio
async def test_internal_plugin_event_waits_behind_debounced_user_input():
    adapter = _RoutingAdapter()
    entry = _entry()
    source = entry.origin
    assert source is not None
    session_key = entry.session_key
    state = SimpleNamespace(
        turn=SimpleNamespace(agent=MagicMock()),
        conversation=SimpleNamespace(queued_events=[]),
    )
    runner = _runner(entry, adapter)
    runner._peek_session_state = lambda _key: state
    runner._session_state = lambda _key: state
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter.set_message_handler(AsyncMock())
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    user_event = MessageEvent(text="human input", source=source)
    adapter._text_debounce[session_key] = TextDebounceState(
        event=user_event,
        task=None,
        first_ts=0.0,
        last_ts=0.0,
    )
    adapter_event = MessageEvent(
        text="plugin wake",
        source=source,
        internal=True,
        allow_gateway_control=False,
        metadata={"gateway_session_key": session_key},
    )

    await adapter.handle_message(adapter_event)

    assert adapter._pending_messages[session_key] is user_event
    assert state.conversation.queued_events == [adapter_event]


@pytest.mark.asyncio
async def test_base_adapter_rejects_derived_session_mismatch():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    event = MessageEvent(
        text="ordinary input",
        source=_entry().origin,
        internal=True,
        allow_gateway_control=False,
        metadata={"gateway_session_key": "agent:main:telegram:dm:other"},
    )

    await adapter.handle_message(event)

    adapter._message_handler.assert_not_awaited()
    assert adapter._active_sessions == {}


@pytest.mark.asyncio
async def test_scheduler_submits_dispatch_on_live_gateway_loop():
    runner = _runner(_entry())
    runner._gateway_loop = asyncio.get_running_loop()
    runner._dispatch_plugin_message_injection = AsyncMock(return_value=True)

    assert (
        runner._schedule_plugin_message_injection(
            session_key="agent:main:telegram:dm:42",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is True
    )

    await asyncio.sleep(0)
    runner._dispatch_plugin_message_injection.assert_awaited_once_with(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
        private=False,
    )


@pytest.mark.asyncio
async def test_scheduler_ignores_same_loop_task_cancellation():
    runner = _runner(_entry())
    loop = asyncio.get_running_loop()
    runner._gateway_loop = loop
    callback_errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: callback_errors.append(context))

    blocker = asyncio.Event()

    async def _wait_for_cancellation(**_kwargs):
        await blocker.wait()

    runner._dispatch_plugin_message_injection = _wait_for_cancellation

    try:
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is True
        )

        task = next(iter(runner._background_tasks))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert callback_errors == []


@pytest.mark.asyncio
async def test_scheduler_logs_async_failure_without_callback_error(caplog):
    runner = _runner(_entry())
    loop = asyncio.get_running_loop()
    runner._gateway_loop = loop
    callback_errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: callback_errors.append(context))
    runner._dispatch_plugin_message_injection = AsyncMock(
        side_effect=RuntimeError("adapter failed")
    )

    try:
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is True
        )
        task = next(iter(runner._background_tasks))
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert callback_errors == []
    assert "plugin=notify-plugin session=key" in caplog.text


def test_scheduler_uses_threadsafe_bridge_outside_gateway_loop():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop

    def _submit(coro, target_loop, **_kwargs):
        assert target_loop is loop
        coro.close()
        future = concurrent.futures.Future()
        future.set_result(True)
        return future

    with patch("gateway.run.safe_schedule_threadsafe", side_effect=_submit) as submit:
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is True
        )

    submit.assert_called_once()


def test_scheduler_ignores_threadsafe_future_cancellation():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop

    def _submit(coro, _target_loop, **_kwargs):
        coro.close()
        future = concurrent.futures.Future()
        future.cancel()
        return future

    with (
        patch("gateway.run.safe_schedule_threadsafe", side_effect=_submit),
        patch("gateway.run.logger.warning") as warning,
    ):
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is True
        )

    warning.assert_not_called()


def test_scheduler_rejects_stopped_or_closed_gateway():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop
    runner._running = False

    assert (
        runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is False
    )
    loop.call_soon_threadsafe.assert_not_called()

    runner._running = True
    runner._gateway_loop = None
    assert (
        runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is False
    )

    runner._gateway_loop = loop
    loop.is_closed.return_value = True
    assert (
        runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is False
    )
    loop.call_soon_threadsafe.assert_not_called()


def test_scheduler_rejects_submission_failure():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop

    def _reject(coro, _target_loop, **_kwargs):
        coro.close()
        return None

    with patch("gateway.run.safe_schedule_threadsafe", side_effect=_reject):
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is False
        )


def test_install_and_clear_gateway_injector_preserves_newer_owner():
    runner = _runner(_entry())
    manager = PluginManager()

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner._install_plugin_message_injector()
        assert manager.has_gateway_message_injector is True

        runner._clear_plugin_message_injector()
        assert manager.has_gateway_message_injector is False

        runner._install_plugin_message_injector()

        newer_owner = MagicMock()
        newer_injector = MagicMock(return_value=True)
        manager.set_gateway_message_injector(newer_owner, newer_injector)
        runner._clear_plugin_message_injector()

    assert manager.has_gateway_message_injector is True
    assert manager.inject_gateway_message(value="kept") is True
    newer_injector.assert_called_once_with(value="kept")


@pytest.mark.asyncio
async def test_gateway_ready_emits_once_after_all_live_gateway_seams():
    """Ready is deferred until running, loop, injector, and idle checker exist."""
    runner = object.__new__(GatewayRunner)
    runner._gateway_ready_emitted = False
    runner._running = False
    runner._gateway_loop = asyncio.get_running_loop()
    manager = PluginManager()

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        with patch("hermes_cli.lifecycle.invoke_hook") as invoke_hook:
            runner._emit_gateway_ready_once()
            invoke_hook.assert_not_called()

            runner._running = True
            runner._emit_gateway_ready_once()
            invoke_hook.assert_not_called()

            manager.set_gateway_message_injector(runner, MagicMock())
            runner._emit_gateway_ready_once()
            invoke_hook.assert_not_called()

            manager.set_gateway_session_idle_checker(runner, lambda _key: True)
            runner._emit_gateway_ready_once()
            runner._emit_gateway_ready_once()

    invoke_hook.assert_called_once_with("gateway_ready", gateway=runner)


@pytest.mark.asyncio
async def test_ordinary_user_input_queues_behind_private_turn_with_cooldown_ack():
    adapter = _RoutingAdapter()
    adapter._send_with_retry = AsyncMock(return_value=None)
    entry = _entry()
    runner = _runner(entry, adapter)
    session_key = entry.session_key
    private_agent = MagicMock()
    state = SimpleNamespace(
        turn=SimpleNamespace(
            private_turn=True,
            agent=private_agent,
            busy_ack_ts=0.0,
            started_ts=0.0,
        )
    )
    runner._peek_session_state = lambda _key: state
    runner._session_state = lambda _key: state
    runner._adapter_for_source = lambda _source: adapter
    runner._effective_busy_input_mode = lambda _source: "interrupt"
    runner._effective_busy_text_mode = lambda _source: "interrupt"
    runner._queue_or_replace_pending_event = MagicMock(
        side_effect=lambda key, event: adapter._pending_messages.__setitem__(key, event)
    )
    runner._reply_anchor_for_event = lambda _event: None
    runner._thread_metadata_for_source = lambda *_args: None

    first = MessageEvent(text="first queued message", source=entry.origin)
    second = MessageEvent(text="second queued message", source=entry.origin)

    assert await runner._handle_active_session_busy_message(first, session_key) is True
    assert adapter._pending_messages[session_key] is first
    private_agent.interrupt.assert_not_called()
    private_agent.steer.assert_not_called()
    adapter._send_with_retry.assert_awaited_once()
    assert adapter._send_with_retry.await_args.kwargs["content"] == (
        "Private turn in progress — your message is queued for the next turn."
    )

    assert await runner._handle_active_session_busy_message(second, session_key) is True
    assert adapter._pending_messages[session_key] is second
    assert adapter._send_with_retry.await_count == 1


def test_plugin_idle_probe_sees_adapter_pending_and_debounced_user_input():
    adapter = _RoutingAdapter()
    entry = _entry()
    runner = _runner(entry, adapter)
    session_key = entry.session_key
    source = entry.origin
    assert source is not None
    state = SimpleNamespace(
        turn=SimpleNamespace(agent=None),
        conversation=SimpleNamespace(queued_events=[]),
    )
    runner._peek_session_state = lambda session_key: state

    assert runner._is_plugin_session_idle(session_key) is True

    adapter._pending_messages[session_key] = MessageEvent(
        text="human follow-up", source=source
    )
    assert runner._is_plugin_session_idle(session_key) is False
    adapter._pending_messages.clear()

    adapter._text_debounce[session_key] = TextDebounceState(
        event=MessageEvent(text="debounced human input", source=source),
        task=None,
        first_ts=0.0,
        last_ts=0.0,
    )
    assert runner._is_plugin_session_idle(session_key) is False
    adapter._text_debounce.clear()

    state.conversation.queued_events.append(MessageEvent(text="queued", source=source))
    assert runner._is_plugin_session_idle(session_key) is False


def test_turn_state_clear_resets_private_plugin_turn_metadata():
    state = TurnState(
        agent=MagicMock(),
        private_turn=True,
        busy_ack="queued notice",
        busy_ack_ts=123.0,
    )

    state.clear()

    assert state.agent is None
    assert state.private_turn is False
    assert state.busy_ack is None
    assert state.busy_ack_ts == 0.0


@pytest.mark.asyncio
async def test_plugin_dispatch_receipt_reports_route_acceptance():
    runner = _runner(_entry())
    runner._gateway_loop = asyncio.get_running_loop()
    runner._dispatch_plugin_message_injection = AsyncMock(return_value=True)
    receipt = MagicMock()

    assert runner._schedule_plugin_message_injection(
        session_key="key",
        content="wake up",
        plugin_id="notify-plugin",
        on_dispatch_result=receipt,
    ) is True

    task = next(iter(runner._background_tasks))
    await task
    receipt.assert_called_once_with(True)


@pytest.mark.asyncio
async def test_plugin_dispatch_receipt_reports_downstream_rejection():
    runner = _runner(_entry())
    runner._gateway_loop = asyncio.get_running_loop()
    runner._dispatch_plugin_message_injection = AsyncMock(return_value=False)
    receipt = MagicMock()

    assert runner._schedule_plugin_message_injection(
        session_key="key",
        content="wake up",
        plugin_id="notify-plugin",
        on_dispatch_result=receipt,
    ) is True

    task = next(iter(runner._background_tasks))
    await task
    receipt.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_plugin_dispatch_receipt_reports_downstream_exception():
    runner = _runner(_entry())
    runner._gateway_loop = asyncio.get_running_loop()
    runner._dispatch_plugin_message_injection = AsyncMock(
        side_effect=RuntimeError("route failed")
    )
    receipt = MagicMock()

    assert runner._schedule_plugin_message_injection(
        session_key="key",
        content="wake up",
        plugin_id="notify-plugin",
        on_dispatch_result=receipt,
    ) is True

    task = next(iter(runner._background_tasks))
    await asyncio.gather(task, return_exceptions=True)
    receipt.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_plugin_dispatch_receipt_failure_is_isolated():
    runner = _runner(_entry())
    runner._gateway_loop = asyncio.get_running_loop()
    runner._dispatch_plugin_message_injection = AsyncMock(return_value=True)
    receipt = MagicMock(side_effect=RuntimeError("receipt failed"))
    loop_errors = []
    previous_handler = asyncio.get_running_loop().get_exception_handler()
    asyncio.get_running_loop().set_exception_handler(
        lambda _loop, context: loop_errors.append(context)
    )

    try:
        assert runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
            on_dispatch_result=receipt,
        ) is True
        task = next(iter(runner._background_tasks))
        await task
        await asyncio.sleep(0)
    finally:
        asyncio.get_running_loop().set_exception_handler(previous_handler)

    receipt.assert_called_once_with(True)
    assert loop_errors == []


def test_plugin_dispatch_receipt_reports_stale_gateway_rejection():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop
    runner._running = False
    receipt = MagicMock()

    assert runner._schedule_plugin_message_injection(
        session_key="key",
        content="wake up",
        plugin_id="notify-plugin",
        on_dispatch_result=receipt,
    ) is False

    receipt.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_plugin_dispatch_rechecks_pinned_session_at_adapter_acceptance():
    """A route switch after the runner lookup must be reported as rejected."""
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    old_entry = _entry()
    new_entry = _entry()
    new_entry.session_id = "session-43"
    runner = _runner(old_entry, adapter)
    adapter.gateway_runner = runner
    lookups = iter([old_entry, old_entry, new_entry])
    runner._async_session_store.lookup_by_session_key = AsyncMock(
        side_effect=lambda _key: next(lookups)
    )

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=old_entry.session_key,
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter._message_handler.assert_not_awaited()


def test_scheduler_rejects_unstarted_unclosed_gateway_loop_without_coroutine():
    runner = _runner(_entry())
    loop = asyncio.new_event_loop()
    receipt = MagicMock()
    runner._gateway_loop = loop

    try:
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
                on_dispatch_result=receipt,
            )
            is False
        )
        receipt.assert_called_once_with(False)
        assert asyncio.all_tasks(loop) == set()
        assert runner._background_tasks == set()
    finally:
        loop.close()
