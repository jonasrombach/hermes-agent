"""Tests for plugin-triggered turns in existing gateway sessions."""

import asyncio
import concurrent.futures
import threading
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
)
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
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
async def test_private_plugin_event_reaches_gateway_setup_before_early_persistence(
    monkeypatch,
):
    """Exercise the plugin event -> gateway cache -> turn-start write boundary.

    Earlier coverage manually set ``agent._gateway_private_turn`` on an already
    constructed agent. That could not detect a lost marker while the real
    plugin event crossed the gateway's cached/new-agent setup.
    """
    from gateway.run import TurnRunner, _is_private_turn_event
    from gateway.turn_context import TurnContext
    from run_agent import AIAgent as RealAIAgent

    adapter = SimpleNamespace(handle_message=AsyncMock())
    dispatch_runner = _runner(_entry(), adapter)
    assert await dispatch_runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="PRIVATE injected envelope",
        plugin_id="ambient-wake",
        private=True,
    )
    event = adapter.handle_message.await_args.args[0]
    assert _is_private_turn_event(event, event.source) is True

    early_persistence = []
    constructed = []

    class ProbeAgent:
        def __init__(self, **kwargs):
            constructed.append(kwargs["session_id"])
            self.model = kwargs["model"]
            self.session_id = kwargs["session_id"]
            self.tools = []
            self.context_compressor = SimpleNamespace(
                last_prompt_tokens=0,
                context_length=200_000,
            )
            self.session_prompt_tokens = 0
            self.session_completion_tokens = 0
            self._gateway_private_turn = False
            self._persist_user_message_idx = 1
            self._session_persist_lock = None
            self._session_db = None
            self._drop_trailing_empty_response_scaffolding = lambda _messages: None
            self._save_session_log = lambda _messages: (_ for _ in ()).throw(
                AssertionError("private turn wrote a session log")
            )
            self._flush_messages_to_session_db = lambda *_args: (_ for _ in ()).throw(
                AssertionError("private turn wrote canonical state.db")
            )

        def run_conversation(self, message, conversation_history=None, **_kwargs):
            messages = list(conversation_history or []) + [
                {"role": "user", "content": message}
            ]
            RealAIAgent._persist_session(self, messages, conversation_history)
            early_persistence.append(
                (self._gateway_private_turn, list(self._session_messages))
            )
            return {"final_response": "NO_REPLY", "messages": messages, "api_calls": 1}

    gateway_runner = MagicMock()
    gateway_runner.config = SimpleNamespace(streaming=None)
    gateway_runner._provider_routing = {}
    gateway_runner._agent_cache_lock = threading.RLock()
    gateway_runner._agent_cache = {}
    gateway_runner._session_db = None
    gateway_runner._prefill_messages = None
    gateway_runner._pending_model_notes = {}
    gateway_runner._pending_skills_reload_notes = {}
    gateway_runner.session_store._entries = {}
    gateway_runner._get_system_prompt_for_channel.return_value = None
    gateway_runner._resolve_session_agent_runtime.return_value = ("test-model", {})
    gateway_runner._resolve_session_reasoning_config.return_value = None
    gateway_runner._resolve_session_service_tier.return_value = None
    gateway_runner._resolve_turn_agent_config.return_value = {
        "model": "test-model", "runtime": {}
    }
    gateway_runner._agent_config_signature.return_value = ("test-signature",)
    gateway_runner._extract_cache_busting_config.return_value = {}
    gateway_runner._refresh_fallback_model.return_value = None
    gateway_runner._consume_pending_native_image_paths.return_value = []
    gateway_runner._consume_pending_turn_sidecar_notes.return_value = []
    gateway_runner._is_telegram_topic_lane.return_value = False
    gateway_runner._is_discord_auto_thread_lane.return_value = False
    gateway_runner._is_relay_discord_channel_lane.return_value = False

    ctx = TurnContext(
        source=event.source,
        message=event.text,
        history=[{"role": "assistant", "content": "ordinary history"}],
        session_id="session-42",
        session_key="agent:main:telegram:dm:42",
        private_turn=_is_private_turn_event(event, event.source),
        user_config={},
        AIAgent=ProbeAgent,
        resolve_display_setting=lambda *_args: False,
        _run_still_current=lambda: True,
        _hooks_ref=SimpleNamespace(loaded_hooks=False),
    )

    monkeypatch.setattr("agent.agent_runtime_helpers.note_turn_persisted", lambda _agent: None)
    TurnRunner(gateway_runner, ctx).run_sync()
    # Reuse the cached agent too: the turn-private bit must be reset from the
    # trusted event context before *every* early persistence boundary.
    TurnRunner(gateway_runner, ctx).run_sync()

    assert constructed == ["session-42"]
    assert early_persistence == [
        (True, [{"role": "assistant", "content": "ordinary history"}]),
        (True, [{"role": "assistant", "content": "ordinary history"}]),
    ]


def test_private_turn_marker_survives_telegram_topic_source_recovery():
    """The authoritative event marker survives dataclasses.replace() recovery."""
    from gateway.run import _restore_private_turn_source_after_recovery

    entry = _entry()
    source = entry.origin
    assert source is not None
    event = MessageEvent(
        text="private wake",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        metadata={"hermes_private_turn": True},
    )
    recovered = _restore_private_turn_source_after_recovery(
        event,
        source,
        thread_id="recovered-topic",
    )

    assert recovered.thread_id == "recovered-topic"
    assert event.source is recovered
    assert getattr(recovered, "_hermes_private_turn") is True
    assert event.metadata["hermes_private_turn"] is True


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
async def test_dispatch_rechecks_pinned_session_at_adapter_acceptance():
    adapter = SimpleNamespace(handle_message=AsyncMock(return_value=True))
    original = _entry()
    moved = _entry()
    moved.session_id = "session-43"
    runner = _runner(original, adapter)
    runner._async_session_store.lookup_by_session_key = AsyncMock(
        side_effect=[original, moved]
    )
    runner._gateway_loop = asyncio.get_running_loop()
    receipt = MagicMock()

    assert runner._schedule_plugin_message_injection(
        session_key=original.session_key,
        content="wake up",
        plugin_id="notify-plugin",
        on_dispatch_result=receipt,
    ) is True
    task = next(iter(runner._background_tasks))
    accepted = await task

    assert accepted is False
    adapter.handle_message.assert_not_awaited()
    receipt.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_dispatch_rejects_in_place_session_id_change_before_adapter_acceptance():
    adapter = SimpleNamespace(handle_message=AsyncMock(return_value=True))
    entry = _entry()
    runner = _runner(entry, adapter)
    lookup_count = 0

    async def _lookup(_session_key):
        nonlocal lookup_count
        lookup_count += 1
        if lookup_count == 2:
            entry.session_id = "session-43"
        return entry

    runner._async_session_store.lookup_by_session_key = _lookup
    runner._gateway_loop = asyncio.get_running_loop()
    receipt = MagicMock()

    assert runner._schedule_plugin_message_injection(
        session_key=entry.session_key,
        content="wake up",
        plugin_id="notify-plugin",
        on_dispatch_result=receipt,
    ) is True
    task = next(iter(runner._background_tasks))
    accepted = await task

    assert accepted is False
    adapter.handle_message.assert_not_awaited()
    receipt.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_plugin_dispatch_receipt_reports_missing_routing_once():
    runner = _runner(None, SimpleNamespace(handle_message=AsyncMock()))
    runner._gateway_loop = asyncio.get_running_loop()
    receipt = MagicMock()

    assert runner._schedule_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
        on_dispatch_result=receipt,
    ) is True
    task = next(iter(runner._background_tasks))
    assert await task is False

    receipt.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_plugin_dispatch_receipt_reports_stale_routing_once():
    original = _entry()
    runner = _runner(original, SimpleNamespace(handle_message=AsyncMock()))
    runner._async_session_store.lookup_by_session_key = AsyncMock(
        side_effect=[original, None]
    )
    runner._gateway_loop = asyncio.get_running_loop()
    receipt = MagicMock()

    assert runner._schedule_plugin_message_injection(
        session_key=original.session_key,
        content="wake up",
        plugin_id="notify-plugin",
        on_dispatch_result=receipt,
    ) is True
    task = next(iter(runner._background_tasks))
    assert await task is False

    receipt.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_plugin_dispatch_receipt_reports_authorization_failure_once():
    runner = _runner(_entry(), SimpleNamespace(handle_message=AsyncMock()))
    runner._is_user_authorized.return_value = False
    runner._gateway_loop = asyncio.get_running_loop()
    receipt = MagicMock()

    assert runner._schedule_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
        on_dispatch_result=receipt,
    ) is True
    task = next(iter(runner._background_tasks))
    assert await task is False

    receipt.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_dispatch_rejects_changed_origin_before_adapter_acceptance():
    adapter = SimpleNamespace(handle_message=AsyncMock(return_value=True))
    original = _entry()
    moved = _entry()
    assert moved.origin is not None
    moved.origin.chat_id = "99"
    runner = _runner(original, adapter)
    runner._async_session_store.lookup_by_session_key = AsyncMock(
        side_effect=[original, moved]
    )

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=original.session_key,
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_rejects_in_place_origin_change_before_adapter_acceptance():
    adapter = SimpleNamespace(handle_message=AsyncMock(return_value=True))
    entry = _entry()
    runner = _runner(entry, adapter)
    lookup_count = 0

    async def _lookup(_session_key):
        nonlocal lookup_count
        lookup_count += 1
        if lookup_count == 2:
            entry.origin.chat_id = "99"
        return entry

    runner._async_session_store.lookup_by_session_key = _lookup

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key,
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
async def test_private_plugin_event_waits_for_turn_boundary_without_busy_handler():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    source = _entry().origin
    session_key = build_session_key(source)
    adapter._active_sessions[session_key] = asyncio.Event()
    busy_handler = AsyncMock(return_value=True)
    adapter.set_busy_session_handler(busy_handler)
    event = MessageEvent(
        text="private wake",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        allow_gateway_control=False,
        metadata={
            "gateway_session_key": session_key,
            "gateway_session_strict": True,
            "hermes_private_turn": True,
        },
    )

    assert await adapter.handle_message(event) is True

    busy_handler.assert_not_awaited()
    adapter._message_handler.assert_not_awaited()
    assert adapter._pending_private_messages[session_key] is event
    assert session_key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_ordinary_pending_event_drains_before_private_plugin_event():
    adapter = _RoutingAdapter()
    source = _entry().origin
    session_key = build_session_key(source)
    ordinary = MessageEvent(text="ordinary", source=source)
    private = MessageEvent(
        text="private wake",
        source=source,
        internal=True,
        allow_gateway_control=False,
        metadata={"hermes_private_turn": True},
    )
    adapter._pending_messages[session_key] = ordinary
    adapter._pending_private_messages[session_key] = private

    assert adapter.get_pending_message(session_key) is ordinary
    assert adapter.get_pending_message(session_key) is private
    assert adapter.get_pending_message(session_key) is None


@pytest.mark.asyncio
async def test_second_private_plugin_event_is_rejected_without_merging_or_busy_ack():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)
    runner._gateway_loop = asyncio.get_running_loop()
    adapter._active_sessions[entry.session_key] = asyncio.Event()
    first_receipt = MagicMock()
    second_receipt = MagicMock()

    assert runner._schedule_plugin_message_injection(
        session_key=entry.session_key,
        content="first private wake",
        plugin_id="notify-plugin",
        private=True,
        on_dispatch_result=first_receipt,
    ) is True
    first_task = next(iter(runner._background_tasks))
    assert await first_task is True

    assert runner._schedule_plugin_message_injection(
        session_key=entry.session_key,
        content="second private wake",
        plugin_id="notify-plugin",
        private=True,
        on_dispatch_result=second_receipt,
    ) is True
    second_task = next(task for task in runner._background_tasks if task is not first_task)
    assert await second_task is False

    assert adapter._pending_private_messages[entry.session_key].text == "first private wake"
    first_receipt.assert_called_once_with(True)
    second_receipt.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_waiting_private_event_dispatches_with_private_and_strict_metadata():
    adapter = _RoutingAdapter()
    adapter.config.typing_indicator = False
    source = _entry().origin
    session_key = build_session_key(source)
    current_started = asyncio.Event()
    release_current = asyncio.Event()
    seen = []

    async def handler(event):
        seen.append(event)
        if event.text == "current turn":
            current_started.set()
            await release_current.wait()
        return None

    adapter.set_message_handler(handler)
    await adapter.handle_message(MessageEvent(text="current turn", source=source))
    await asyncio.wait_for(current_started.wait(), timeout=1)
    private = MessageEvent(
        text="private wake",
        source=source,
        internal=True,
        allow_gateway_control=False,
        metadata={
            "hermes_private_turn": True,
            "gateway_session_key": session_key,
            "gateway_session_id": "session-42",
            "gateway_session_strict": True,
        },
    )

    assert await adapter.handle_message(private) is True
    release_current.set()
    for _ in range(100):
        if len(seen) == 2 and session_key not in adapter._active_sessions:
            break
        await asyncio.sleep(0.01)

    assert [event.text for event in seen] == ["current turn", "private wake"]
    assert seen[1] is private
    assert seen[1].metadata["hermes_private_turn"] is True
    assert seen[1].metadata["gateway_session_strict"] is True
    assert seen[1].metadata["gateway_session_id"] == "session-42"


@pytest.mark.asyncio
async def test_sequential_private_turn_releases_latches_without_overlap():
    adapter = _RoutingAdapter()
    adapter.config.typing_indicator = False
    source = _entry().origin
    session_key = build_session_key(source)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    running = 0
    max_running = 0
    seen = []

    async def handler(event):
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        seen.append(event.text)
        try:
            if event.text == "first":
                first_started.set()
                await release_first.wait()
        finally:
            running -= 1
        return None

    adapter.set_message_handler(handler)
    await adapter.handle_message(MessageEvent(text="first", source=source))
    await asyncio.wait_for(first_started.wait(), timeout=1)
    private = MessageEvent(
        text="private second",
        source=source,
        internal=True,
        allow_gateway_control=False,
        metadata={"hermes_private_turn": True},
    )
    assert await adapter.handle_message(private) is True
    release_first.set()
    for _ in range(100):
        if seen == ["first", "private second"] and session_key not in adapter._active_sessions:
            break
        await asyncio.sleep(0.01)

    assert seen == ["first", "private second"]
    assert max_running == 1
    assert session_key not in adapter._active_sessions
    assert session_key not in adapter._session_tasks


@pytest.mark.asyncio
async def test_non_private_busy_event_keeps_existing_pending_slot_behavior():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    source = _entry().origin
    session_key = build_session_key(source)
    adapter._active_sessions[session_key] = asyncio.Event()
    event = MessageEvent(
        text="ordinary plugin event",
        source=source,
        internal=True,
        allow_gateway_control=False,
        metadata={"gateway_session_key": session_key},
    )

    await adapter.handle_message(event)

    assert adapter._pending_messages[session_key] is event
    assert session_key not in adapter._pending_private_messages


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
    )


@pytest.mark.asyncio
async def test_plugin_dispatch_receipt_reports_adapter_acceptance_once():
    adapter = SimpleNamespace(handle_message=AsyncMock(return_value=True))
    entry = _entry()
    runner = _runner(entry, adapter)
    runner._gateway_loop = asyncio.get_running_loop()
    receipt = MagicMock()

    assert runner._schedule_plugin_message_injection(
        session_key=entry.session_key,
        content="wake up",
        plugin_id="notify-plugin",
        on_dispatch_result=receipt,
    ) is True

    task = next(iter(runner._background_tasks))
    await task
    assert receipt.call_args_list == [((True,), {})]


@pytest.mark.asyncio
async def test_plugin_dispatch_receipt_reports_adapter_rejection_once():
    adapter = SimpleNamespace(handle_message=AsyncMock(return_value=False))
    entry = _entry()
    runner = _runner(entry, adapter)
    runner._gateway_loop = asyncio.get_running_loop()
    receipt = MagicMock()

    assert runner._schedule_plugin_message_injection(
        session_key=entry.session_key,
        content="wake up",
        plugin_id="notify-plugin",
        on_dispatch_result=receipt,
    ) is True

    task = next(iter(runner._background_tasks))
    await task
    assert receipt.call_args_list == [((False,), {})]


@pytest.mark.asyncio
async def test_plugin_dispatch_receipt_reports_cancellation_once():
    runner = _runner(_entry())
    runner._gateway_loop = asyncio.get_running_loop()
    blocked = asyncio.Event()

    async def _blocked_dispatch(**_kwargs):
        await blocked.wait()

    runner._dispatch_plugin_message_injection = _blocked_dispatch
    receipt = MagicMock()

    assert runner._schedule_plugin_message_injection(
        session_key="key",
        content="wake up",
        plugin_id="notify-plugin",
        on_dispatch_result=receipt,
    ) is True

    task = next(iter(runner._background_tasks))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert receipt.call_args_list == [((False,), {})]


@pytest.mark.asyncio
async def test_plugin_dispatch_receipt_callback_exception_is_isolated():
    runner = _runner(_entry())
    runner._gateway_loop = asyncio.get_running_loop()
    runner._dispatch_plugin_message_injection = AsyncMock(return_value=True)
    receipt = MagicMock(side_effect=RuntimeError("receipt failed"))

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
async def test_plugin_dispatch_receipt_reports_adapter_exception_once():
    adapter = SimpleNamespace(
        handle_message=AsyncMock(side_effect=RuntimeError("adapter failed"))
    )
    runner = _runner(_entry(), adapter)
    runner._gateway_loop = asyncio.get_running_loop()
    receipt = MagicMock()

    assert runner._schedule_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
        on_dispatch_result=receipt,
    ) is True
    task = next(iter(runner._background_tasks))
    assert await task is False

    receipt.assert_called_once_with(False)


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


def test_scheduler_receipt_reports_submission_failure_once():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop
    receipt = MagicMock()

    def _reject(coro, _target_loop, **_kwargs):
        coro.close()
        return None

    with patch("gateway.run.safe_schedule_threadsafe", side_effect=_reject):
        assert runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
            on_dispatch_result=receipt,
        ) is False

    receipt.assert_called_once_with(False)


def test_scheduler_receipt_reports_submission_exception_once():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop
    receipt = MagicMock()

    def _raise(coro, _target_loop, **_kwargs):
        coro.close()
        raise RuntimeError("loop bridge failed")

    with patch("gateway.run.safe_schedule_threadsafe", side_effect=_raise):
        assert runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
            on_dispatch_result=receipt,
        ) is False

    receipt.assert_called_once_with(False)


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
