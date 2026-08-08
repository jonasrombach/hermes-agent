"""Behavior tests for session-wake follow-up turn boundaries."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter():
    adapter = _StubAdapter(
        PlatformConfig(enabled=True, token="test"),
        Platform.TELEGRAM,
    )
    adapter._send_with_retry = AsyncMock(return_value=None)
    return adapter


def _event(
    text: str,
    *,
    wake_key: str | None = None,
    ambient_marker: str = "session_wake",
    expires_at: str | None = None,
) -> MessageEvent:
    metadata = {}
    internal = wake_key is not None
    if wake_key is not None:
        metadata = {ambient_marker: True, "coalesce_key": wake_key}
        if expires_at is not None:
            metadata["expires_at"] = expires_at
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="42",
            chat_type="dm",
        ),
        internal=internal,
        metadata=metadata,
    )


def _session_key() -> str:
    return build_session_key(_event("key").source)


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    async def _poll():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout=timeout)


@pytest.mark.asyncio
@pytest.mark.parametrize("ambient_marker", ["session_wake", "internal_ambient"])
async def test_trusted_ambient_markers_share_queue_priority(ambient_marker):
    adapter = _make_adapter()
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(event):
        if event.text == "active":
            started.set()
            await release.wait()
        return ""

    adapter._message_handler = handler
    await adapter.handle_message(_event("active"))
    await started.wait()
    await adapter.handle_message(
        _event("ambient", wake_key="key", ambient_marker=ambient_marker)
    )
    await adapter.handle_message(_event("human"))
    assert adapter._pending_messages[_session_key()].text == "human"
    assert [e.text for e in adapter._session_wake_queues[_session_key()]] == ["ambient"]
    release.set()
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_busy_session_wake_uses_separate_queue_without_busy_ack():
    adapter = _make_adapter()
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(event):
        started.set()
        await release.wait()
        return ""

    adapter._message_handler = handler
    adapter._busy_session_handler = AsyncMock(return_value=False)

    await adapter.handle_message(_event("active"))
    await started.wait()
    await adapter.handle_message(_event("wake", wake_key="heartbeat"))
    await adapter.handle_message(_event("human"))

    session_key = _session_key()
    assert adapter._pending_messages[session_key].text == "human"
    assert [event.text for event in adapter._session_wake_queues[session_key]] == ["wake"]
    adapter._busy_session_handler.assert_awaited_once()
    assert adapter._busy_session_handler.await_args.args[0].text == "human"

    release.set()
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_human_follow_up_runs_before_ordered_session_wakes():
    adapter = _make_adapter()
    started = asyncio.Event()
    release = asyncio.Event()
    processed = []

    async def handler(event):
        processed.append(event.text)
        if event.text == "active":
            started.set()
            await release.wait()
        return ""

    adapter._message_handler = handler
    await adapter.handle_message(_event("active"))
    await started.wait()
    await adapter.handle_message(_event("wake-a", wake_key="a"))
    await adapter.handle_message(_event("wake-b", wake_key="b"))
    await adapter.handle_message(_event("human"))

    release.set()
    await _wait_until(lambda: len(processed) == 4)

    assert processed == ["active", "human", "wake-a", "wake-b"]
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_same_key_session_wakes_collapse_to_newest_without_reordering_keys():
    adapter = _make_adapter()
    started = asyncio.Event()
    release = asyncio.Event()
    processed = []

    async def handler(event):
        processed.append(event.text)
        if event.text == "active":
            started.set()
            await release.wait()
        return ""

    adapter._message_handler = handler
    await adapter.handle_message(_event("active"))
    await started.wait()
    await adapter.handle_message(_event("wake-a-old", wake_key="a"))
    await adapter.handle_message(_event("wake-b", wake_key="b"))
    await adapter.handle_message(_event("wake-a-new", wake_key="a"))

    release.set()
    await _wait_until(lambda: len(processed) == 3)

    assert processed == ["active", "wake-a-new", "wake-b"]
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_active_same_key_session_wake_keeps_only_newest_follow_up():
    adapter = _make_adapter()
    started = asyncio.Event()
    release = asyncio.Event()
    processed = []

    async def handler(event):
        processed.append(event.text)
        if event.text == "wake-active":
            started.set()
            await release.wait()
        return ""

    adapter._message_handler = handler
    await adapter.handle_message(_event("wake-active", wake_key="heartbeat"))
    await started.wait()
    await adapter.handle_message(_event("wake-old", wake_key="heartbeat"))
    await adapter.handle_message(_event("wake-newest", wake_key="heartbeat"))

    session_key = _session_key()
    assert [event.text for event in adapter._session_wake_queues[session_key]] == [
        "wake-newest"
    ]

    release.set()
    await _wait_until(lambda: processed == ["wake-active", "wake-newest"])
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_empty_coalesce_keys_remain_distinct():
    adapter = _make_adapter()
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(event):
        if event.text == "active":
            started.set()
            await release.wait()
        return ""

    adapter._message_handler = handler
    await adapter.handle_message(_event("active"))
    await started.wait()
    await adapter.handle_message(_event("wake-a", wake_key=""))
    await adapter.handle_message(_event("wake-b", wake_key=""))

    session_key = _session_key()
    assert [event.text for event in adapter._session_wake_queues[session_key]] == [
        "wake-a",
        "wake-b",
    ]

    release.set()
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_idle_session_wake_starts_immediately():
    adapter = _make_adapter()
    processed = []

    async def handler(event):
        processed.append(event.text)
        return ""

    adapter._message_handler = handler
    await adapter.handle_message(_event("wake", wake_key="heartbeat"))
    await _wait_until(lambda: processed == ["wake"])

    assert _session_key() not in adapter._session_wake_queues
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_expired_ambient_wake_is_dropped_when_dequeued():
    adapter = _make_adapter()
    started = asyncio.Event()
    release = asyncio.Event()
    processed = []

    async def handler(event):
        processed.append(event.text)
        if event.text == "active":
            started.set()
            await release.wait()
        return ""

    adapter._message_handler = handler
    await adapter.handle_message(_event("active"))
    await started.wait()
    await adapter.handle_message(
        _event(
            "expired-while-queued",
            wake_key="ambient",
            ambient_marker="internal_ambient",
            expires_at=(datetime.now(timezone.utc) + timedelta(milliseconds=20)).isoformat(),
        )
    )
    await asyncio.sleep(0.05)
    release.set()
    await _wait_until(lambda: _session_key() not in adapter._active_sessions)

    assert processed == ["active"]
    assert _session_key() not in adapter._session_wake_queues
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_stale_lock_heal_discards_session_wakes():
    adapter = _make_adapter()
    session_key = _session_key()
    done_task = asyncio.create_task(asyncio.sleep(0))
    await done_task
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._session_tasks[session_key] = done_task
    adapter._session_wake_queues[session_key] = [_event("stale", wake_key="a")]
    adapter._message_handler = AsyncMock(return_value="")

    await adapter.handle_message(_event("fresh"))

    assert session_key not in adapter._session_wake_queues
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_cancel_with_discard_pending_discards_session_wakes():
    adapter = _make_adapter()
    session_key = _session_key()
    adapter._session_wake_queues[session_key] = [_event("queued", wake_key="a")]

    await adapter.cancel_session_processing(session_key, discard_pending=True)

    assert session_key not in adapter._session_wake_queues


@pytest.mark.asyncio
async def test_reset_discards_queued_session_wakes():
    adapter = _make_adapter()
    started = asyncio.Event()
    processed = []

    async def handler(event):
        processed.append(event.text)
        if event.text == "active":
            started.set()
            await asyncio.Event().wait()
        return ""

    adapter._message_handler = handler
    await adapter.handle_message(_event("active"))
    await started.wait()
    await adapter.handle_message(_event("queued", wake_key="a"))
    await adapter.handle_message(_event("/reset"))

    assert _session_key() not in adapter._session_wake_queues
    assert processed == ["active", "/reset"]
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_wake_racing_active_reset_is_rejected():
    adapter = _make_adapter()
    adapter._message_handler = AsyncMock(return_value="")
    session_key = _session_key()
    active_task = asyncio.create_task(asyncio.Event().wait())
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._session_tasks[session_key] = active_task
    adapter._background_tasks.add(active_task)
    adapter._discarding_session_wakes.add(session_key)

    with pytest.raises(RuntimeError, match="reset is in progress"):
        await adapter.handle_message(_event("wake", wake_key="heartbeat"))

    active_task.cancel()
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_wake_arriving_during_idle_reset_is_rejected_and_not_drained():
    adapter = _make_adapter()
    reset_started = asyncio.Event()
    release_reset = asyncio.Event()
    processed = []

    async def handler(event):
        processed.append(event.text)
        if event.text == "/reset":
            reset_started.set()
            await release_reset.wait()
        return ""

    adapter._message_handler = handler
    await adapter.handle_message(_event("/reset"))
    await reset_started.wait()

    with pytest.raises(RuntimeError, match="reset is in progress"):
        await adapter.handle_message(_event("wake", wake_key="heartbeat"))

    release_reset.set()
    await _wait_until(lambda: processed == ["/reset"])
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_shutdown_discards_and_rejects_new_session_wakes():
    adapter = _make_adapter()
    adapter._session_wake_queues[_session_key()] = [_event("queued", wake_key="a")]
    handler = AsyncMock(return_value="")
    adapter._message_handler = handler

    await adapter.cancel_background_tasks()

    assert adapter._session_wake_queues == {}
    with pytest.raises(RuntimeError, match="shutting down"):
        await adapter.handle_message(_event("wake-after-shutdown", wake_key="a"))
    handler.assert_not_awaited()
