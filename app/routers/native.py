"""Native ``/transcribe`` endpoint with richer options.

Differences from the OpenAI-compatible endpoint:

- Explicit ``timestamps`` and ``vad`` toggles.
- Always returns JSON with metadata (duration, segments, words).
- Supports the same SRT/VTT response formats.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from ..asr import Segment
from ..audio import decode_audio
from ..config import get_settings
from ..schemas import (
    NativeTranscribeResponse,
    OpenAIResponseFormat,
    SegmentTimestamp,
    WordTimestamp,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["native"])


@router.post("/transcribe")
async def transcribe(
    request: Request,
    file: Annotated[UploadFile, File(description="Audio file to transcribe")],
    timestamps: Annotated[bool, Form()] = False,
    vad: Annotated[bool | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    response_format: Annotated[OpenAIResponseFormat, Form()] = "json",
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

    result = await asr.transcribe(
        decoded.waveform,
        sample_rate=decoded.sample_rate,
        with_timestamps=timestamps,
        use_vad=vad,
    )

    logger.info(
        "native transcribe: duration=%.2fs inference=%.2fs vad=%s timestamps=%s chars=%d",
        result.duration_s,
        result.inference_s,
        vad,
        timestamps,
        len(result.text),
    )

    if response_format == "text":
        return PlainTextResponse(result.text)

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

    payload = NativeTranscribeResponse(
        text=result.text,
        duration=result.duration_s,
        language=language or result.language,
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


# ---------------------------------------------------------------------------
# Subtitle formatting helpers (also reused by routers/openai.py)
# ---------------------------------------------------------------------------


def _format_timestamp(seconds: float, *, vtt: bool) -> str:
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    sep = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def _segments_or_fallback(
    segments: list[Segment],
    fallback_text: str,
    duration: float,
) -> list[Segment]:
    if segments:
        return segments
    if not fallback_text.strip():
        return []
    end = max(duration, 0.1)
    return [Segment(start=0.0, end=end, text=fallback_text.strip())]


def _format_srt(segments: list[Segment], *, fallback_text: str, duration: float) -> str:
    items = _segments_or_fallback(segments, fallback_text, duration)
    out: list[str] = []
    for i, seg in enumerate(items, start=1):
        out.append(str(i))
        out.append(
            f"{_format_timestamp(seg.start, vtt=False)} --> "
            f"{_format_timestamp(seg.end, vtt=False)}"
        )
        out.append(seg.text)
        out.append("")
    return "\n".join(out)


def _format_vtt(segments: list[Segment], *, fallback_text: str, duration: float) -> str:
    items = _segments_or_fallback(segments, fallback_text, duration)
    out: list[str] = ["WEBVTT", ""]
    for seg in items:
        out.append(
            f"{_format_timestamp(seg.start, vtt=True)} --> "
            f"{_format_timestamp(seg.end, vtt=True)}"
        )
        out.append(seg.text)
        out.append("")
    return "\n".join(out)
