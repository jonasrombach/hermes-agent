import asyncio
from unittest.mock import MagicMock, patch

import pytest

from gateway.run import GatewayRunner
from hermes_cli.plugins import PluginManager, VALID_HOOKS


def test_gateway_ready_is_a_supported_plugin_hook():
    assert "gateway_ready" in VALID_HOOKS


@pytest.mark.asyncio
async def test_gateway_ready_is_emitted_once_per_runner():
    runner = object.__new__(GatewayRunner)
    runner._gateway_ready_emitted = False
    runner._running = True
    runner._gateway_loop = asyncio.get_running_loop()
    manager = PluginManager()
    manager.set_gateway_message_injector(runner, MagicMock())
    manager.set_gateway_session_idle_checker(runner, MagicMock())
    hook = MagicMock()

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        with patch("hermes_cli.lifecycle.invoke_hook", hook):
            runner._emit_gateway_ready_once()
            runner._emit_gateway_ready_once()

    hook.assert_called_once_with("gateway_ready", gateway=runner)


@pytest.mark.asyncio
async def test_gateway_ready_hook_failure_is_isolated():
    runner = object.__new__(GatewayRunner)
    runner._gateway_ready_emitted = False
    runner._running = True
    runner._gateway_loop = asyncio.get_running_loop()
    manager = PluginManager()
    manager.set_gateway_message_injector(runner, MagicMock())
    manager.set_gateway_session_idle_checker(runner, MagicMock())

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        with patch(
            "hermes_cli.lifecycle.invoke_hook",
            side_effect=RuntimeError("plugin failed"),
        ):
            runner._emit_gateway_ready_once()

    assert runner._gateway_ready_emitted is True
