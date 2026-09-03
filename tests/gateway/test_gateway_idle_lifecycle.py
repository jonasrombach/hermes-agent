"""Lifecycle regression tests for gateway/plugin idle accounting."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="reply")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _make_runner(adapter: BasePlatformAdapter, raw_key: str, recovered_key: str, recovered_source: SessionSource):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        platforms={Platform.TELEGRAM: adapter.config},
        sessions_dir="/tmp",
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._running = True
    runner._draining = False
    runner._background_tasks = set()
    runner._is_user_authorized = MagicMock(return_value=True)
    runner._gateway_loop = None
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._claim_active_session_slot = MagicMock(return_value=(None, None))
    runner._persist_active_agents = MagicMock()
    runner._clear_durable_active_turn = AsyncMock()
    runner._run_post_turn_hooks = AsyncMock()
    runner._clear_restart_failure_count = AsyncMock()
    runner._adapter_for_source = MagicMock(return_value=adapter)
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._is_session_run_current = MagicMock(return_value=True)
    runner._session_key_for_source = MagicMock(
        side_effect=lambda source: (
            recovered_key
            if getattr(source, "thread_id", None) == recovered_source.thread_id
            else raw_key
        )
    )

    async def handle_turn(event, _source, _quick_key, _generation):
        # This is the state transition performed by normal Telegram topic
        # recovery + agent publication: the event source is rewritten by
        # _handle_message_with_agent, while _run_agent publishes the live agent
        # under the recovered session key.
        event.source = recovered_source
        runner._session_state(recovered_key).turn.agent = object()
        return "reply"

    runner._handle_message_with_agent = AsyncMock(side_effect=handle_turn)
    return runner


@pytest.mark.asyncio
async def test_two_recovered_telegram_turns_release_every_idle_latch():
    raw_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="dm",
        user_id="42",
    )
    recovered_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="dm",
        user_id="42",
        thread_id="topic-9",
    )
    raw_key = "agent:main:telegram:dm:42"
    recovered_key = "agent:main:telegram:dm:42:thread:topic-9"

    adapter = _Adapter(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True, message_id="reply"))
    runner = _make_runner(adapter, raw_key, recovered_key, recovered_source)
    adapter.set_message_handler(runner._handle_message)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    for text in ("first", "second"):
        assert await adapter.handle_message(
            MessageEvent(text=text, message_type=MessageType.TEXT, source=raw_source)
        ) is True
        task = adapter._session_tasks[raw_key]
        await task

    recovered_state = runner._peek_session_state(recovered_key)
    assert recovered_state is not None
    assert recovered_state.turn.agent is None
    assert raw_key not in adapter._active_sessions
    assert raw_key not in adapter._pending_messages
    assert raw_key not in adapter._text_debounce
    assert runner._is_plugin_session_idle(recovered_key) is True

    await adapter.cancel_background_tasks()
