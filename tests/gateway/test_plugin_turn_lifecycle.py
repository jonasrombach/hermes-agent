import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import MessageEvent
from tests.gateway.test_plugin_message_injection import _RoutingAdapter, _entry, _runner


@pytest.mark.asyncio
async def test_injected_turn_reports_start_and_finish_only_when_executed():
    adapter = _RoutingAdapter()
    adapter.config.typing_indicator = False
    gate = asyncio.Event()
    started = asyncio.Event()

    async def handler(event):
        started.set()
        await gate.wait()

    adapter.set_message_handler(handler)
    runner = _runner(_entry(), adapter)
    states = []
    accepted = await runner._dispatch_plugin_message_injection(
        session_key=_entry().session_key, content="wake", plugin_id="test",
        private=True, on_turn_state=states.append,
    )
    assert accepted
    await started.wait()
    assert states == ["started"]
    gate.set()
    await asyncio.gather(*list(adapter._background_tasks))
    assert states == ["started", "finished"]


@pytest.mark.asyncio
async def test_discarded_pending_private_turn_reports_cancelled_once():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    entry = _entry()
    states = []
    event = await _queued_private_event(adapter, entry.session_key, states)

    await adapter.cancel_session_processing(entry.session_key)

    assert states == ["cancelled"]
    assert entry.session_key not in adapter._pending_private_messages
    assert event not in adapter._pending_private_messages.get(entry.session_key, ())


@pytest.mark.asyncio
async def test_reset_discards_private_wakes_but_preserves_ordinary_follow_up():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    entry = _entry()
    states = []
    await _queued_private_event(adapter, entry.session_key, states)
    ordinary = MessageEvent(text="ordinary follow-up", source=entry.origin)
    adapter._pending_messages[entry.session_key] = ordinary

    await adapter.cancel_session_processing(
        entry.session_key,
        release_guard=False,
        discard_pending=False,
    )

    assert states == ["cancelled"]
    assert adapter._pending_messages[entry.session_key] is ordinary
    assert entry.session_key not in adapter._pending_private_messages


@pytest.mark.asyncio
async def test_failing_injected_turn_reports_cancelled_not_finished():
    adapter = _RoutingAdapter()
    adapter.config.typing_indicator = False
    entry = _entry()
    states = []

    async def handler(_event):
        raise RuntimeError("turn failed")

    adapter.set_message_handler(handler)
    runner = _runner(entry, adapter)
    assert await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key,
        content="wake",
        plugin_id="test",
        private=True,
        on_turn_state=states.append,
    )

    await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)
    assert states == ["started", "cancelled"]


@pytest.mark.asyncio
async def test_failing_private_turn_never_delivers_error_detail():
    adapter = _RoutingAdapter()
    adapter.config.typing_indicator = False
    adapter.send = AsyncMock()
    entry = _entry()
    states = []

    async def handler(_event):
        raise RuntimeError("private wake failure detail")

    adapter.set_message_handler(handler)
    runner = _runner(entry, adapter)
    assert await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key,
        content="wake",
        plugin_id="test",
        private=True,
        on_turn_state=states.append,
    )

    await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)
    adapter.send.assert_not_awaited()
    assert states == ["started", "cancelled"]


async def _queued_private_event(adapter, session_key, states):
    source = _entry().origin
    adapter._active_sessions[session_key] = asyncio.Event()
    event = MessageEvent(
        text="private wake",
        source=source,
        internal=True,
        allow_gateway_control=False,
        metadata={"hermes_private_turn": True, "gateway_session_key": session_key},
    )
    setattr(event, "_injected_turn_state_callback", states.append)
    assert await adapter.handle_message(event) is True
    return event
