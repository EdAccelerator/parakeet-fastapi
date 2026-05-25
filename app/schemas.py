"""Pydantic request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# OpenAI uses these literal response_format values; we also add SRT/VTT.
OpenAIResponseFormat = Literal["json", "text", "verbose_json", "srt", "vtt"]


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class SegmentTimestamp(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    """Minimal OpenAI-compatible response for response_format=json."""

    text: str


class VerboseTranscriptionResponse(BaseModel):
    """OpenAI-style verbose response for response_format=verbose_json."""

    task: Literal["transcribe"] = "transcribe"
    language: str | None = None
    duration: float | None = None
    text: str
    segments: list[SegmentTimestamp] = Field(default_factory=list)
    words: list[WordTimestamp] = Field(default_factory=list)


class NativeTranscribeResponse(BaseModel):
    """Richer native response with timestamps and metadata."""

    text: str
    duration: float | None = None
    language: str | None = None
    segments: list[SegmentTimestamp] = Field(default_factory=list)
    words: list[WordTimestamp] = Field(default_factory=list)


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "nvidia"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model_loaded: bool
    model_name: str | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    """Matches OpenAI's error envelope shape for compatibility."""

    error: "ErrorBody"


class ErrorBody(BaseModel):
    message: str
    type: str
    code: str | None = None


ErrorResponse.model_rebuild()
