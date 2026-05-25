"""Liveness/readiness probes and model listing."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from ..config import get_settings
from ..schemas import HealthResponse, ModelInfo, ModelList

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(request: Request) -> JSONResponse:
    asr = getattr(request.app.state, "asr", None)
    loaded = bool(getattr(asr, "is_loaded", False))
    settings = get_settings()
    body = HealthResponse(
        status="ok" if loaded else "loading",
        model_loaded=loaded,
        model_name=settings.parakeet_model_name if loaded else None,
        detail=None if loaded else "Model is still loading",
    )
    code = status.HTTP_200_OK if loaded else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=body.model_dump())


@router.get("/v1/models", response_model=ModelList, tags=["openai"])
async def list_models() -> ModelList:
    settings = get_settings()
    return ModelList(
        data=[
            ModelInfo(
                id=settings.parakeet_model_name,
                owned_by="nvidia",
                created=int(time.time()),
            )
        ]
    )


@router.get("/", tags=["meta"])
async def root() -> dict:
    settings = get_settings()
    return {
        "service": "parakeet-fastapi",
        "model": settings.parakeet_model_name,
        "quantization": settings.parakeet_quantization,
        "endpoints": {
            "openai_transcriptions": "POST /v1/audio/transcriptions",
            "native_transcribe": "POST /transcribe",
            "models": "GET /v1/models",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }
