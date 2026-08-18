from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner
from hermes_cli.plugins import PluginState


class Event:
    source = None

    def __init__(self, args: str):
        self._args = args

    def get_command_args(self) -> str:
        return self._args


@pytest.fixture
def adaptive_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state = PluginState("rocky-heartbeat")
    state.set(
        "heartbeat",
        {
            "enabled": True,
            "next_at": "2026-08-18T11:45:00+00:00",
            "last_completed_at": "2026-08-18T11:00:00+00:00",
            "runs_completed": 2,
            "history": [
                {
                    "run_id": "run-2",
                    "completed_at": "2026-08-18T11:00:00+00:00",
                    "next_at": "2026-08-18T11:45:00+00:00",
                    "notify": False,
                    "message": "",
                    "note": "Checked calendar and mail; nothing moved, stayed silent.",
                }
            ],
        },
    )
    return state


@pytest.mark.asyncio
async def test_heartbeat_status_prefers_rocky_adaptive_state(adaptive_state):
    response = await GatewayRunner._handle_heartbeat_command(
        SimpleNamespace(), Event("status")
    )

    assert "🖤 Rocky Heartbeat" in response
    assert "Di., 18.08.2026 · 13:45 CEST" in response
    assert "Runs: 2" in response


@pytest.mark.asyncio
async def test_heartbeat_last_shows_structured_audit_record(adaptive_state):
    response = await GatewayRunner._handle_heartbeat_command(
        SimpleNamespace(), Event("last")
    )

    assert "Checked calendar and mail" in response
    assert "**Entscheidung**\nstill" in response
    assert "**Nächster Lauf**\n" in response
    assert response.startswith("🖤 **Letzter Heartbeat**\n\n")
    assert "\n\n**Prüfprotokoll**\n" in response


@pytest.mark.asyncio
async def test_heartbeat_history_uses_separated_readable_entries(adaptive_state):
    response = await GatewayRunner._handle_heartbeat_command(
        SimpleNamespace(), Event("history")
    )

    assert response.startswith("🖤 **Heartbeat-Verlauf**\n\n")
    assert "**1 · Di., 18.08.2026 · 13:00 CEST**" in response
    assert "\n• Entscheidung: still\n• Nächster Lauf:" in response
    assert "\n\nChecked calendar and mail" in response


@pytest.mark.asyncio
async def test_heartbeat_last_separates_sent_message(adaptive_state):
    state = adaptive_state.get("heartbeat")
    state["history"][0]["notify"] = True
    state["history"][0]["message"] = "Visible message"
    adaptive_state.set("heartbeat", state)

    response = await GatewayRunner._handle_heartbeat_command(
        SimpleNamespace(), Event("last")
    )

    assert "\n\n**Gesendete Nachricht**\nVisible message" in response
