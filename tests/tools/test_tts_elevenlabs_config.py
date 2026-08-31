"""Regression tests for ElevenLabs request configuration."""

from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools import tts_streaming, tts_tool


@pytest.fixture(autouse=True)
def _fake_optional_elevenlabs_types(monkeypatch):
    """Keep these unit tests independent of the optional premium SDK."""
    package = ModuleType("elevenlabs")
    types_module = ModuleType("elevenlabs.types")
    setattr(types_module, "VoiceSettings", lambda **kwargs: SimpleNamespace(**kwargs))
    setattr(package, "types", types_module)
    monkeypatch.setitem(__import__("sys").modules, "elevenlabs", package)
    monkeypatch.setitem(__import__("sys").modules, "elevenlabs.types", types_module)


def test_elevenlabs_forwards_speed_and_text_normalization(tmp_path):
    client = MagicMock()
    client.text_to_speech.convert.return_value = iter([b"audio"])

    with (
        patch.object(tts_tool, "_resolve_provider_key", return_value="test-key"),
        patch.object(tts_tool, "_import_elevenlabs") as import_elevenlabs,
    ):
        import_elevenlabs.return_value = MagicMock(return_value=client)
        output = str(tmp_path / "voice.ogg")
        tts_tool._generate_elevenlabs(
            "Version 11.15",
            output,
            {"elevenlabs": {"speed": 1.15, "apply_text_normalization": "on"}},
        )

    kwargs = client.text_to_speech.convert.call_args.kwargs
    assert kwargs["voice_settings"].speed == 1.15
    assert kwargs["apply_text_normalization"] == "on"


def test_elevenlabs_streaming_forwards_options(monkeypatch):
    client = MagicMock()
    client.text_to_speech.convert.return_value = iter([b"audio"])
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: MagicMock(return_value=client))
    monkeypatch.setattr(tts_streaming, "_resolve_key", lambda *_args: "test-key")

    provider = tts_streaming.ElevenLabsStreamer(
        {}, {"speed": 1.15, "apply_text_normalization": "on"}
    )
    assert list(provider.stream("Version 11.15")) == [b"audio"]

    kwargs = client.text_to_speech.convert.call_args.kwargs
    assert kwargs["voice_settings"].speed == 1.15
    assert kwargs["apply_text_normalization"] == "on"


@pytest.mark.parametrize("speed", [0.69, 1.21, "fast"])
def test_elevenlabs_rejects_invalid_speed(speed):
    with pytest.raises(ValueError, match="tts.elevenlabs.speed"):
        tts_tool._elevenlabs_convert_options({"speed": speed})


def test_elevenlabs_rejects_invalid_normalization():
    with pytest.raises(ValueError, match="apply_text_normalization"):
        tts_tool._elevenlabs_convert_options({"apply_text_normalization": "yes"})