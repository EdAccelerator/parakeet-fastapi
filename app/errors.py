"""Centralised exception handlers and error envelopes."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .audio import AudioDecodeError, AudioTooLongError

logger = logging.getLogger(__name__)


def _envelope(message: str, type_: str, code: str | None = None) -> dict:
    return {"error": {"message": message, "type": type_, "code": code}}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AudioDecodeError)
    async def _on_decode_err(_req: Request, exc: AudioDecodeError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_envelope(str(exc), "invalid_request_error", "audio_decode_error"),
        )

    @app.exception_handler(AudioTooLongError)
    async def _on_too_long(_req: Request, exc: AudioTooLongError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=_envelope(str(exc), "invalid_request_error", "audio_too_long"),
        )

    @app.exception_handler(RequestValidationError)
    async def _on_validation(_req: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_envelope(
                "Request validation failed: " + str(exc.errors()),
                "invalid_request_error",
                "validation_error",
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _on_http(_req: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(
                exc.detail if isinstance(exc.detail, str) else str(exc.detail),
                "http_error",
                None,
            ),
        )

    @app.exception_handler(Exception)
    async def _on_unhandled(_req: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception in request")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope(
                f"Internal server error: {exc.__class__.__name__}",
                "internal_error",
                None,
            ),
        )
