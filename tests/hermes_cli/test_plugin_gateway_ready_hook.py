from hermes_cli.plugins import VALID_HOOKS


def test_gateway_ready_is_a_supported_plugin_hook():
    assert "gateway_ready" in VALID_HOOKS