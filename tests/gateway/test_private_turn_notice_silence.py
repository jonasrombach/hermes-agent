"""Private gateway turns do not emit automatic notice rails."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


class _NoticeAdapter:
    supports_status_text = False

    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, **kwargs):
        self.sent.append(content)
        return SimpleNamespace(success=True, message_id="notice-1")

    async def edit_message(self, *args, **kwargs):
        return SimpleNamespace(success=False)

    def has_pending_interrupt(self, _session_key):
        return False

    def get_pending_message(self, _session_key):
        return None


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="private-notice-chat",
        chat_type="dm",
        user_id="private-notice-user",
    )


def _inner_runner(adapter):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._draining = False
    runner._gateway_loop = asyncio.get_running_loop()
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._get_proxy_url = lambda: None
    runner._resolve_enabled_toolsets_for_source = lambda *_args: None
    runner._adapter_for_source = lambda _source: adapter
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: None
    runner._thread_metadata_for_target = lambda *_args, **_kwargs: None
    runner._is_session_run_current = lambda *_args: True
    runner._release_running_agent_state = lambda *_args, **_kwargs: True
    runner._run_in_executor_with_context = lambda func: asyncio.to_thread(func)
    return runner


def _install_short_agent(monkeypatch, *, duration):
    class Agent:
        model = "test-model"

        def get_activity_summary(self):
            return {
                "seconds_since_activity": 0.01,
                "last_activity_desc": "waiting",
                "current_tool": "terminal",
                "api_call_count": 1,
                "max_iterations": 2,
            }

    def run_sync(turn_runner):
        turn_runner._ctx.agent_holder[0] = Agent()
        time.sleep(duration)
        result = {
            "final_response": "trusted final",
            "messages": [],
            "tools": [],
            "api_calls": 1,
        }
        turn_runner._ctx.result_holder[0] = result
        return result

    monkeypatch.setattr(gateway_run.TurnRunner, "run_sync", run_sync)


async def _run_inner_notice_path(monkeypatch, *, private_turn, duration=0.08):
    adapter = _NoticeAdapter()
    runner = _inner_runner(adapter)
    _install_short_agent(monkeypatch, duration=duration)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    real_wait = asyncio.wait

    async def accelerated_wait(awaitables, *, timeout=None, return_when=asyncio.ALL_COMPLETED):
        return await real_wait(
            awaitables,
            timeout=min(timeout, 0.001) if timeout is not None else None,
            return_when=return_when,
        )

    monkeypatch.setattr(gateway_run.asyncio, "wait", accelerated_wait)
    monkeypatch.setattr(
        gateway_run,
        "_float_env",
        lambda name, default: {
            "HERMES_AGENT_NOTIFY_INTERVAL": 0.001,
            "HERMES_AGENT_TIMEOUT": 10.0,
            "HERMES_AGENT_TIMEOUT_WARNING": 0.001,
        }.get(name, default),
    )

    result = await runner._run_agent_inner(
        "private background work",
        "",
        [],
        _source(),
        "notice-session",
        session_key=None,
        private_turn=private_turn,
    )
    assert result["final_response"] == "trusted final"
    return adapter.sent


@pytest.mark.asyncio
async def test_private_turn_suppresses_long_running_notice(monkeypatch):
    sent = await _run_inner_notice_path(monkeypatch, private_turn=True)

    assert not any("Working" in content for content in sent)


@pytest.mark.asyncio
async def test_ordinary_turn_still_sends_long_running_notice(monkeypatch):
    sent = await _run_inner_notice_path(monkeypatch, private_turn=False)

    assert any("Working" in content for content in sent)


@pytest.mark.asyncio
async def test_private_turn_suppresses_inactivity_warning(monkeypatch):
    sent = await _run_inner_notice_path(monkeypatch, private_turn=True, duration=0.03)

    assert not any("No activity" in content for content in sent)


@pytest.mark.asyncio
async def test_ordinary_turn_still_sends_inactivity_warning(monkeypatch):
    sent = await _run_inner_notice_path(monkeypatch, private_turn=False, duration=0.03)

    assert any("No activity" in content for content in sent)


@pytest.mark.asyncio
async def test_private_turn_suppresses_auto_reset_notice(monkeypatch, tmp_path):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    adapter = _NoticeAdapter()
    source = _source()
    event = MessageEvent(
        text="private wake",
        source=source,
        internal=True,
        metadata={"hermes_private_turn": True},
    )
    entry = SessionEntry(
        session_key="notice-session",
        session_id="notice-session",
        created_at=time.time(),
        updated_at=time.time(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        was_auto_reset=True,
        reset_had_activity=True,
        auto_reset_reason="idle",
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._run_agent = AsyncMock(return_value={
        "final_response": "trusted final",
        "response_transformed": True,
        "messages": [],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
    })
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_args, **_kwargs: 100_000)

    await runner._handle_message_with_agent(event, source, "notice-session", 1)

    assert not any("Session automatically reset" in content for content in adapter.sent)


@pytest.mark.asyncio
async def test_ordinary_turn_still_sends_auto_reset_notice(monkeypatch, tmp_path):
    # This uses the private-path fixture but removes only the private marker.
    # It protects the existing reset notice behavior from the new privacy gate.
    adapter = _NoticeAdapter()
    runner = gateway_run.GatewayRunner(GatewayConfig())
    source = _source()
    event = MessageEvent(text="ordinary wake", source=source, internal=True)
    entry = SessionEntry(
        session_key="notice-session",
        session_id="notice-session",
        created_at=time.time(),
        updated_at=time.time(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        was_auto_reset=True,
        reset_had_activity=True,
        auto_reset_reason="idle",
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._run_agent = AsyncMock(return_value={
        "final_response": "final",
        "messages": [],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
    })
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_args, **_kwargs: 100_000)

    await runner._handle_message_with_agent(event, source, "notice-session", 1)

    assert any("Session automatically reset" in content for content in adapter.sent)
