"""OpenAI-compatible ``/v1/audio/transcriptions`` endpoint.

Implements a subset of the OpenAI Audio API that drop-in clients (openai-python,
LangChain, etc.) actually use. ``model``, ``temperature`` and ``prompt`` are
accepted for compatibility but largely informational — Parakeet TDT uses
greedy decoding and has no prompt conditioning.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from ..audio import decode_audio
from ..config import get_settings
from ..schemas import (
    SegmentTimestamp,
    TranscriptionResponse,
    VerboseTranscriptionResponse,
    WordTimestamp,
)
from .native import _format_srt, _format_vtt  # reuse formatter helpers

logger = logging.getLogger(__name__)
router = APIRouter(tags=["openai"])

ResponseFormat = Literal["json", "text", "verbose_json", "srt", "vtt"]


@router.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: Annotated[UploadFile, File(description="Audio file to transcribe")],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,  # noqa: ARG001 - accepted for compat
    response_format: Annotated[ResponseFormat, Form()] = "json",
    temperature: Annotated[float | None, Form()] = None,  # noqa: ARG001 - accepted for compat
    timestamp_granularities: Annotated[list[str] | None, Form()] = None,
) -> Response:
    settings = get_settings()
    asr = request.app.state.asr
    if asr is None or not asr.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ASR model is not loaded yet",
        )

    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Upload of {len(raw)} bytes exceeds max_upload_bytes="
                f"{settings.max_upload_bytes}"
            ),
        )

    decoded = decode_audio(raw, settings, filename_hint=file.filename)

    want_timestamps = response_format == "verbose_json" or response_format in {"srt", "vtt"}
    if timestamp_granularities:
        want_timestamps = True

    result = await asr.transcribe(
        decoded.waveform,
        sample_rate=decoded.sample_rate,
        with_timestamps=want_timestamps,
    )

    logger.info(
        "transcribed: model=%s language_hint=%s duration=%.2fs inference=%.2fs chars=%d",
        model or settings.parakeet_model_name,
        language,
        result.duration_s,
        result.inference_s,
        len(result.text),
    )

    if response_format == "text":
        return PlainTextResponse(result.text)

    if response_format == "json":
        return JSONResponse(TranscriptionResponse(text=result.text).model_dump())

    if response_format == "verbose_json":
        payload = VerboseTranscriptionResponse(
            language=language,
            duration=result.duration_s,
            text=result.text,
            segments=[
                SegmentTimestamp(start=s.start, end=s.end, text=s.text)
                for s in result.segments
            ],
            words=[
                WordTimestamp(word=w.word, start=w.start, end=w.end)
                for w in result.words
            ],
        )
        return JSONResponse(payload.model_dump())

    if response_format == "srt":
        return PlainTextResponse(
            _format_srt(result.segments, fallback_text=result.text, duration=result.duration_s),
            media_type="application/x-subrip",
        )

    if response_format == "vtt":
        return PlainTextResponse(
            _format_vtt(result.segments, fallback_text=result.text, duration=result.duration_s),
            media_type="text/vtt",
        )

    # Should be unreachable due to Literal validation.
    raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")
