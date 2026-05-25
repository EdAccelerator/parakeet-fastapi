"""Endpoint tests using FastAPI's TestClient + a fake ASR backend.

These exercise routing, request parsing, response shaping and error envelopes
without loading the real ONNX model. The model is replaced with a stub before
the lifespan startup runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient


# We tell the lifespan to skip loading the real model, then install a fake.
os.environ["PARAKEET_SKIP_MODEL_LOAD"] = "1"


@dataclass
class FakeResult:
    text: str
    duration_s: float = 1.0
    inference_s: float = 0.05
    segments: list = field(default_factory=list)
    words: list = field(default_factory=list)
    language: str | None = None


class FakeASR:
    """Minimal stand-in for ParakeetASR."""

    is_loaded = True

    def __init__(self, canned_text: str = "hello world"):
        self.canned_text = canned_text
        self.last_call: dict | None = None

    async def transcribe(self, waveform, *, sample_rate, with_timestamps=False, use_vad=None):
        from app.asr import Segment, Word

        self.last_call = {
            "samples": int(waveform.shape[0]),
            "sample_rate": sample_rate,
            "with_timestamps": with_timestamps,
            "use_vad": use_vad,
        }
        segments: list[Segment] = []
        words: list[Word] = []
        duration = float(waveform.shape[0]) / float(sample_rate) if sample_rate else 0.0
        if with_timestamps:
            segments = [Segment(start=0.0, end=max(duration, 0.1), text=self.canned_text)]
            tokens = self.canned_text.split()
            if tokens:
                step = max(duration, 0.1) / len(tokens)
                words = [
                    Word(word=tok, start=i * step, end=(i + 1) * step)
                    for i, tok in enumerate(tokens)
                ]
        return FakeResult(
            text=self.canned_text,
            duration_s=duration,
            segments=segments,
            words=words,
        )


@pytest.fixture()
def client():
    from app.main import create_app

    app = create_app()
    with TestClient(app) as c:
        # Lifespan ran with model load skipped — install fake now.
        c.app.state.asr = FakeASR()
        yield c


def test_root_returns_service_info(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["service"] == "parakeet-fastapi"
    assert "openai_transcriptions" in data["endpoints"]


def test_health_ok_when_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_503_when_model_not_loaded(client):
    client.app.state.asr = None
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["model_loaded"] is False


def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert any(m["id"] == "parakeet-tdt-0.6b-v3" for m in body["data"])


def test_openai_transcriptions_json_default(client, sine_wav_bytes):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sine.wav", sine_wav_bytes, "audio/wav")},
        data={"model": "parakeet-tdt-0.6b-v3"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"text": "hello world"}


def test_openai_transcriptions_text_response(client, sine_wav_bytes):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sine.wav", sine_wav_bytes, "audio/wav")},
        data={"response_format": "text"},
    )
    assert r.status_code == 200
    assert r.text == "hello world"


def test_openai_transcriptions_verbose_json_includes_segments(client, sine_wav_bytes):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sine.wav", sine_wav_bytes, "audio/wav")},
        data={"response_format": "verbose_json", "language": "en"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "hello world"
    assert body["task"] == "transcribe"
    assert body["language"] == "en"
    assert len(body["segments"]) == 1
    assert body["segments"][0]["text"] == "hello world"
    assert len(body["words"]) == 2  # "hello" + "world"


def test_openai_transcriptions_srt(client, sine_wav_bytes):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sine.wav", sine_wav_bytes, "audio/wav")},
        data={"response_format": "srt"},
    )
    assert r.status_code == 200
    assert "00:00:00,000 -->" in r.text
    assert "hello world" in r.text


def test_openai_transcriptions_vtt(client, sine_wav_bytes):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sine.wav", sine_wav_bytes, "audio/wav")},
        data={"response_format": "vtt"},
    )
    assert r.status_code == 200
    assert r.text.startswith("WEBVTT")


def test_openai_transcriptions_503_when_model_not_loaded(client, sine_wav_bytes):
    client.app.state.asr = None
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sine.wav", sine_wav_bytes, "audio/wav")},
    )
    assert r.status_code == 503


def test_openai_transcriptions_400_on_bad_audio(client):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("garbage.bin", b"this is not audio", "application/octet-stream")},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"


def test_openai_transcriptions_422_without_file(client):
    r = client.post("/v1/audio/transcriptions", data={"model": "x"})
    assert r.status_code == 422


def test_native_transcribe_default_json(client, sine_wav_bytes):
    r = client.post(
        "/transcribe",
        files={"file": ("sine.wav", sine_wav_bytes, "audio/wav")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "hello world"
    assert body["duration"] == pytest.approx(1.0, abs=0.05)


def test_native_transcribe_with_timestamps(client, sine_wav_bytes):
    r = client.post(
        "/transcribe",
        files={"file": ("sine.wav", sine_wav_bytes, "audio/wav")},
        data={"timestamps": "true"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "hello world"
    assert len(body["segments"]) == 1
    assert len(body["words"]) == 2


def test_native_transcribe_records_vad_flag(client, sine_wav_bytes):
    fake = client.app.state.asr
    client.post(
        "/transcribe",
        files={"file": ("sine.wav", sine_wav_bytes, "audio/wav")},
        data={"vad": "false"},
    )
    assert fake.last_call["use_vad"] is False


def test_413_on_oversize_upload(client, monkeypatch, sine_wav_bytes):
    from app import config

    def fake_settings():
        s = config.Settings(
            max_upload_bytes=10,  # tiny limit to force 413
            parakeet_model_dir="/tmp/__nope__",
            parakeet_vad_dir="/tmp/__nope__",
        )
        return s

    monkeypatch.setattr(config, "get_settings", fake_settings)
    # Also patch where the routers imported it.
    import app.routers.openai as oai_router
    import app.routers.native as nat_router

    monkeypatch.setattr(oai_router, "get_settings", fake_settings)
    monkeypatch.setattr(nat_router, "get_settings", fake_settings)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sine.wav", sine_wav_bytes, "audio/wav")},
    )
    assert r.status_code == 413
    body = r.json()
    assert body["error"]["type"] == "http_error"


def test_413_on_audio_too_long(client, monkeypatch, long_sine_wav_bytes):
    """5 s audio with max_audio_seconds=1 should yield 413 from AudioTooLongError."""
    from app import config

    def fake_settings():
        return config.Settings(
            max_audio_seconds=1.0,
            parakeet_model_dir="/tmp/__nope__",
            parakeet_vad_dir="/tmp/__nope__",
        )

    monkeypatch.setattr(config, "get_settings", fake_settings)
    import app.routers.openai as oai_router
    import app.routers.native as nat_router
    import app.audio as audio_mod

    monkeypatch.setattr(oai_router, "get_settings", fake_settings)
    monkeypatch.setattr(nat_router, "get_settings", fake_settings)
    monkeypatch.setattr(audio_mod, "get_settings", fake_settings)

    r = client.post(
        "/transcribe",
        files={"file": ("long.wav", long_sine_wav_bytes, "audio/wav")},
    )
    assert r.status_code == 413
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "exceeds" in body["error"]["message"]
