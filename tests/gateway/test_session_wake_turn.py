"""Focused guards for internal session-wake turn semantics."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource, build_session_key


class _RecordingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append((chat_id, content, metadata))
        return SendResult(success=True, message_id="sent-1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _wake_metadata():
    return {
        "session_wake": True,
        "delivery_id": "heartbeat:2026-08-08T00:00:00Z",
        "coalesce_key": "rocky-heartbeat",
        "display_kind": "hidden",
        "skip_external_memory_sync": True,
    }


def test_internal_session_wake_controls_preserve_hidden_provenance():
    from gateway.run import (
        _session_wake_hook_provenance,
        _turn_controls_from_metadata,
    )

    controls = _turn_controls_from_metadata(_wake_metadata(), internal=True)

    assert controls["persist_user_display_kind"] == "hidden"
    assert controls["persist_user_display_metadata"] == {
        "synthetic": True,
        "source": "session_wake",
        "delivery_id": "heartbeat:2026-08-08T00:00:00Z",
        "coalesce_key": "rocky-heartbeat",
    }
    assert controls["skip_external_memory_sync"] is True

    minimal_controls = _turn_controls_from_metadata(
        {"session_wake": True}, internal=True
    )
    assert minimal_controls["skip_external_memory_sync"] is True
    assert _session_wake_hook_provenance(
        _wake_metadata(), internal=True
    ) == {
        "internal": True,
        "synthetic": True,
        "source": "session_wake",
        "event_kind": "cron_session_wake",
        "delivery_id": "heartbeat:2026-08-08T00:00:00Z",
        "coalesce_key": "rocky-heartbeat",
    }



def test_internal_ambient_controls_match_legacy_session_wake_without_plugin_leakage():
    from gateway.run import _session_wake_hook_provenance, _turn_controls_from_metadata

    metadata = {
        "internal_ambient": True,
        "plugin_id": "weather-plugin",
        "delivery_id": "weather:42",
        "coalesce_key": "plugin:weather-plugin:rain",
        "event_kind": "weather.alert",
        "source_label": "weather station",
    }
    controls = _turn_controls_from_metadata(metadata, internal=True)
    assert controls["persist_user_display_kind"] == "hidden"
    assert controls["skip_external_memory_sync"] is True
    assert controls["persist_user_display_metadata"] == {
        "synthetic": True,
        "source": "ambient",
        "delivery_id": "weather:42",
        "coalesce_key": "plugin:weather-plugin:rain",
    }
    assert _session_wake_hook_provenance(metadata, internal=True) == {
        "internal": True,
        "synthetic": True,
        "source": "ambient",
        "event_kind": "weather.alert",
        "delivery_id": "weather:42",
        "coalesce_key": "plugin:weather-plugin:rain",
        "source_label": "weather station",
    }

    from gateway.run import _turn_controls_from_metadata

    assert _turn_controls_from_metadata(
        {"skip_external_memory_sync": True}, internal=False
    ) == {
        "persist_user_display_kind": None,
        "persist_user_display_metadata": None,
        "skip_external_memory_sync": False,
    }


def test_skip_external_memory_sync_blocks_only_automatic_turn_sync():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.session_id = "session-1"
    agent._skip_external_memory_sync_for_turn = True

    agent._sync_external_memory_for_turn(
        original_user_message="synthetic heartbeat",
        final_response="NO_REPLY",
        interrupted=False,
        messages=[{"role": "user", "content": "synthetic heartbeat"}],
    )

    agent._memory_manager.sync_all.assert_not_called()
    agent._memory_manager.queue_prefetch_all.assert_not_called()
    # The boundary is post-turn automatic sync only. Recall and explicit tools
    # remain exposed on the same manager and are not disabled or replaced.
    assert callable(agent._memory_manager.prefetch_all)
    assert callable(agent._memory_manager.execute_tool)


def test_normal_turn_still_syncs_external_memory():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.session_id = "session-1"
    agent._skip_external_memory_sync_for_turn = False

    messages = [{"role": "user", "content": "remember this substantive request"}]
    agent._sync_external_memory_for_turn(
        original_user_message="remember this substantive request",
        final_response="I will remember it.",
        interrupted=False,
        messages=messages,
    )

    agent._memory_manager.sync_all.assert_called_once_with(
        "remember this substantive request",
        "I will remember it.",
        session_id="session-1",
        messages=messages,
    )
    agent._memory_manager.queue_prefetch_all.assert_called_once_with(
        "remember this substantive request",
        session_id="session-1",
    )


@pytest.mark.asyncio
async def test_internal_session_wake_exact_no_reply_sends_nothing():
    adapter = _RecordingAdapter()
    adapter.config.typing_indicator = True
    adapter._keep_typing = AsyncMock()

    async def handler(_event):
        from gateway.response_filters import is_intentional_silence_agent_result

        agent_result = {"final_response": "NO_REPLY", "messages": []}
        response = agent_result["final_response"]
        if is_intentional_silence_agent_result(agent_result, response):
            response = ""
        return response

    adapter.set_message_handler(handler)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", chat_type="dm")
    event = MessageEvent(
        text="synthetic heartbeat",
        source=source,
        internal=True,
        metadata=_wake_metadata(),
    )

    await adapter._process_message_background(event, build_session_key(source))

    assert adapter.sent == []
    adapter._keep_typing.assert_not_awaited()


@pytest.mark.asyncio
async def test_internal_session_wake_agent_error_is_logged_not_sent():
    adapter = _RecordingAdapter()

    async def handler(_event):
        raise RuntimeError("synthetic wake failed")

    adapter.set_message_handler(handler)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", chat_type="dm")
    event = MessageEvent(
        text="synthetic heartbeat",
        source=source,
        internal=True,
        metadata=_wake_metadata(),
    )

    await adapter._process_message_background(event, build_session_key(source))

    assert adapter.sent == []
