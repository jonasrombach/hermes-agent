"""Tests for async gateway lifecycle hooks exposed to plugins."""

import asyncio

import pytest

from hermes_cli.plugins import VALID_HOOKS, PluginManager


@pytest.mark.asyncio
async def test_invoke_hook_async_awaits_sync_and_async_callbacks():
    manager = PluginManager()
    seen = []

    def sync_hook(**kwargs):
        seen.append(("sync", kwargs["value"]))
        return "sync-result"

    async def async_hook(**kwargs):
        seen.append(("async", kwargs["value"]))
        return "async-result"

    manager._hooks["gateway_startup"] = [sync_hook, async_hook]

    results = await manager.invoke_hook_async("gateway_startup", value=7)

    assert seen == [("sync", 7), ("async", 7)]
    assert results == ["sync-result", "async-result"]


@pytest.mark.asyncio
async def test_invoke_hook_async_isolates_callback_failures():
    manager = PluginManager()

    async def broken(**_kwargs):
        raise RuntimeError("boom")

    async def healthy(**_kwargs):
        return "ok"

    manager._hooks["gateway_shutdown"] = [broken, healthy]

    assert await manager.invoke_hook_async("gateway_shutdown") == ["ok"]



@pytest.mark.asyncio
async def test_gateway_lifecycle_tasks_cancel_and_await_services():
    from hermes_cli.lifecycle import GatewayLifecycleTasks

    observed = []

    async def service():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            observed.append("cancelled")
            raise

    tasks = GatewayLifecycleTasks()
    tasks.create_task(service())
    await __import__("asyncio").sleep(0)
    await tasks.cancel_and_wait()
    assert observed == ["cancelled"]
    assert tasks.create_task is not None

    assert "gateway_startup" in VALID_HOOKS
    assert "gateway_shutdown" in VALID_HOOKS
