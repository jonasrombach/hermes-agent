from types import SimpleNamespace

import pytest

from tools import tts_streaming, tts_tool


class FakeVoiceSettings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeElevenLabsClient:
    def __init__(self):
        self.convert_kwargs = None
        self.text_to_speech = SimpleNamespace(convert=self.convert)

    def convert(self, **kwargs):
        self.convert_kwargs = kwargs
        return iter([b"audio"])


def test_normal_elevenlabs_request_forwards_speed_and_auto_normalization(
    monkeypatch, tmp_path
):
    client = FakeElevenLabsClient()
    monkeypatch.setattr(tts_tool, "_resolve_provider_key", lambda *_: "test-key")
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: lambda **_: client)
    monkeypatch.setattr(
        tts_tool,
        "_import_elevenlabs_voice_settings",
        lambda: FakeVoiceSettings,
        raising=False,
    )
    config = {
        "elevenlabs": {
            "voice_id": "voice",
            "model_id": "model",
            "speed": 1.2,
            "apply_text_normalization": "auto",
        }
    }

    output = tmp_path / "speech.mp3"
    tts_tool._generate_elevenlabs("Hallo 123", str(output), config)

    assert client.convert_kwargs is not None
    assert client.convert_kwargs["apply_text_normalization"] == "auto"
    assert client.convert_kwargs["voice_settings"].kwargs == {"speed": 1.2}
    assert output.read_bytes() == b"audio"


def test_streaming_elevenlabs_request_forwards_speed_and_auto_normalization(
    monkeypatch,
):
    client = FakeElevenLabsClient()
    monkeypatch.setattr(tts_streaming, "_resolve_key", lambda *_: "test-key")
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: lambda **_: client)
    monkeypatch.setattr(
        tts_tool,
        "_import_elevenlabs_voice_settings",
        lambda: FakeVoiceSettings,
    )
    config = {
        "provider": "elevenlabs",
        "elevenlabs": {
            "voice_id": "voice",
            "model_id": "model",
            "speed": 1.2,
            "apply_text_normalization": "auto",
        },
    }
    streamer = tts_streaming.ElevenLabsStreamer(config, config["elevenlabs"])

    assert list(streamer.stream("Hallo 123")) == [b"audio"]
    assert client.convert_kwargs is not None
    assert client.convert_kwargs["apply_text_normalization"] == "auto"
    assert client.convert_kwargs["voice_settings"].kwargs == {"speed": 1.2}


def test_elevenlabs_request_rejects_unknown_text_normalization_value():
    with pytest.raises(ValueError, match="apply_text_normalization"):
        tts_tool._elevenlabs_convert_options(
            {"elevenlabs": {"apply_text_normalization": "sometimes"}}
        )
