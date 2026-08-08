"""Public plugin ambient-turn delivery boundaries."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


@pytest.fixture
def source() -> dict[str, str]:
    return {
        "platform": "telegram",
        "chat_id": "chat-test-1",
        "chat_type": "dm",
        "user_id": "chat-test-1",
    }


@pytest.fixture(autouse=True)
def clear_shared_wake_runtime():
    from gateway.wake import clear_wake_runtime

    clear_wake_runtime()
    yield
    clear_wake_runtime()


class _Runner:
    def __init__(self, adapter, *, secondary=None):
        self.adapters = {Platform.TELEGRAM: adapter}
        self._profile_adapters = secondary or {}
        self._running = True
        self._draining = False


@pytest.mark.asyncio
async def test_ambient_service_delivers_explicit_source_with_bounded_provenance(
    monkeypatch, source
):
    from gateway.ambient_delivery import AmbientTurnService
    from gateway.wake import set_wake_runtime

    adapter = object()
    deliver = AsyncMock()
    monkeypatch.setattr("gateway.ambient_delivery.deliver_wake", deliver)
    set_wake_runtime(_Runner(adapter), asyncio.get_running_loop())

    await AmbientTurnService("weather-plugin").deliver(
        source=source,
        text="Rain begins shortly.",
        event_kind="weather.alert",
        delivery_id="weather-42",
        coalesce_key="weather:rain",
        source_label="weather station",
    )

    args, kwargs = deliver.await_args
    assert args == (adapter,)
    assert kwargs["source"].chat_id == "chat-test-1"
    assert kwargs["metadata"] == {
        "internal_ambient": True,
        "plugin_id": "weather-plugin",
        "delivery_id": "weather-42",
        "coalesce_key": "plugin:weather-plugin:weather:rain",
        "event_kind": "weather.alert",
        "source_label": "weather station",
    }


@pytest.mark.asyncio
async def test_ambient_service_routes_secondary_profile_without_primary_fallback(
    monkeypatch, source
):
    from gateway.ambient_delivery import AmbientTurnService
    from gateway.wake import set_wake_runtime

    primary, secondary = object(), object()
    deliver = AsyncMock()
    monkeypatch.setattr("gateway.ambient_delivery.deliver_wake", deliver)
    set_wake_runtime(
        _Runner(primary, secondary={"work": {Platform.TELEGRAM: secondary}}),
        asyncio.get_running_loop(),
    )

    await AmbientTurnService("plugin").deliver(
        source={**source, "profile": "work"},
        text="event",
        event_kind="sync",
        delivery_id="id-1",
    )
    assert deliver.await_args.args == (secondary,)

    set_wake_runtime(_Runner(primary, secondary={}), asyncio.get_running_loop())
    with pytest.raises(RuntimeError, match="unavailable"):
        await AmbientTurnService("plugin").deliver(
            source={**source, "profile": "work"},
            text="event",
            event_kind="sync",
            delivery_id="id-2",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("text", "", "text"),
        ("text", "x" * 16_385, "16384"),
        ("event_kind", "bad kind", "event_kind"),
        ("delivery_id", "", "delivery_id"),
        ("source_label", "x" * 161, "source_label"),
        ("expires_at", "not-a-time", "expires_at"),
    ],
)
async def test_ambient_service_rejects_malformed_or_oversized_input(
    monkeypatch, source, field, value, message
):
    from gateway.ambient_delivery import AmbientTurnService
    from gateway.wake import set_wake_runtime

    set_wake_runtime(_Runner(object()), asyncio.get_running_loop())
    kwargs = {
        "source": source,
        "text": "event",
        "event_kind": "sync",
        "delivery_id": "id-1",
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        await AmbientTurnService("plugin").deliver(**kwargs)


@pytest.mark.asyncio
async def test_ambient_service_rejects_expired_stopping_or_closed_runtime(source):
    from gateway.ambient_delivery import AmbientTurnService
    from gateway.wake import set_wake_runtime

    loop = asyncio.get_running_loop()
    set_wake_runtime(_Runner(object()), loop)
    with pytest.raises(ValueError, match="expired"):
        await AmbientTurnService("plugin").deliver(
            source=source,
            text="event",
            event_kind="sync",
            delivery_id="expired",
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )

    runner = _Runner(object())
    runner._draining = True
    set_wake_runtime(runner, loop)
    with pytest.raises(RuntimeError, match="stopping"):
        await AmbientTurnService("plugin").deliver(
            source=source, text="event", event_kind="sync", delivery_id="stopping"
        )

    class _ClosedLoop:
        def is_closed(self): return True
        def is_running(self): return False

    with pytest.raises(RuntimeError, match="running event loop"):
        set_wake_runtime(_Runner(object()), _ClosedLoop())


def test_ambient_module_uses_shared_wake_runtime_only():
    import gateway.ambient_delivery as ambient_delivery

    assert not hasattr(ambient_delivery, "set_ambient_runtime")
    assert not hasattr(ambient_delivery, "clear_ambient_runtime")
    assert not hasattr(ambient_delivery, "get_ambient_runtime")


def test_plugin_context_exposes_ambient_service_not_adapter_registry():
    from gateway.ambient_delivery import AmbientTurnService
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    ctx = PluginContext(PluginManifest(name="plugin"), PluginManager())
    assert isinstance(ctx.ambient, AmbientTurnService)
    assert not hasattr(ctx, "adapters")


def test_runtime_resolver_requires_a_real_session_source():
    from gateway.ambient_delivery import resolve_ambient_adapter

    with pytest.raises(ValueError, match="SessionSource"):
        resolve_ambient_adapter(object(), {Platform.TELEGRAM: object()})
