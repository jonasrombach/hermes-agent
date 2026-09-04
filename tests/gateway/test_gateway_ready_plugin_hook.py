"""Lifecycle coverage for the generic gateway_ready plugin hook."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_gateway_ready_fires_once_after_injection_readiness_and_cleans_up_task():
    runner, _adapter = make_restart_runner()
    manager = PluginManager()
    manager._discovered = True
    context = PluginContext(
        PluginManifest(name="ready-plugin", key="ready-plugin", source="user"),
        manager,
    )
    calls: list[str] = []
    task_started = asyncio.Event()
    task_cancelled = asyncio.Event()

    async def background_work() -> None:
        task_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            task_cancelled.set()
            raise

    def failing_hook(**kwargs) -> None:
        calls.append("failing")
        assert kwargs["gateway"] is runner
        assert kwargs["adapters"] is runner.adapters
        raise RuntimeError("isolated plugin failure")

    def task_hook(**kwargs):
        calls.append("task")
        return context.spawn_task(background_work(), name="ready-plugin:background")

    context.register_hook("gateway_ready", failing_hook)
    context.register_hook("gateway_ready", task_hook)

    # Registration only records callbacks; it must not start the plugin task.
    assert calls == []
    assert not task_started.is_set()

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        manager.set_gateway_message_injector(runner, MagicMock(return_value=True))
        runner._fire_gateway_ready_plugin_hooks()
        await asyncio.wait_for(task_started.wait(), timeout=1.0)

        # The lifecycle gate is one-shot, and the returned task is owned by
        # the gateway's standard shutdown task set.
        runner._fire_gateway_ready_plugin_hooks()

    assert calls == ["failing", "task"]
    ready_task = next(
        task for task in runner._background_tasks if task.get_name() == "ready-plugin:background"
    )

    with (
        patch("gateway.status.remove_pid_file"),
        patch("gateway.status.write_runtime_status"),
    ):
        await runner.stop()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(ready_task), timeout=1.0)
    assert task_cancelled.is_set()
