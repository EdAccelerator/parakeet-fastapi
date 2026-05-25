"""Pydantic schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    HealthResponse,
    ModelInfo,
    ModelList,
    NativeTranscribeResponse,
    SegmentTimestamp,
    TranscriptionResponse,
    VerboseTranscriptionResponse,
    WordTimestamp,
)


def test_transcription_response_minimal():
    r = TranscriptionResponse(text="hello world")
    d = r.model_dump()
    assert d == {"text": "hello world"}


def test_verbose_transcription_response_defaults():
    r = VerboseTranscriptionResponse(text="hi")
    assert r.task == "transcribe"
    assert r.segments == []
    assert r.words == []
    assert r.text == "hi"


def test_native_response_with_timestamps():
    r = NativeTranscribeResponse(
        text="hi",
        duration=1.0,
        language="en",
        segments=[SegmentTimestamp(start=0.0, end=0.5, text="hi")],
        words=[WordTimestamp(word="hi", start=0.0, end=0.5)],
    )
    d = r.model_dump()
    assert d["text"] == "hi"
    assert len(d["segments"]) == 1
    assert d["segments"][0]["text"] == "hi"
    assert d["words"][0]["word"] == "hi"


def test_health_response_loading_state():
    r = HealthResponse(status="loading", model_loaded=False, detail="x")
    assert r.status == "loading"
    assert r.model_loaded is False


def test_health_response_rejects_bad_status():
    with pytest.raises(ValidationError):
        HealthResponse(status="weird", model_loaded=True)  # type: ignore[arg-type]


def test_model_list_shape():
    ml = ModelList(data=[ModelInfo(id="parakeet-tdt-0.6b-v3")])
    d = ml.model_dump()
    assert d["object"] == "list"
    assert d["data"][0]["id"] == "parakeet-tdt-0.6b-v3"
    assert d["data"][0]["object"] == "model"
