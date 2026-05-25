"""FastAPI app factory.

The model is loaded once during the lifespan startup hook and disposed on
shutdown. The app is otherwise stateless: no per-request state is retained
across requests, no disk writes other than transient ffmpeg pipes.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .asr import ParakeetASR
from .config import get_settings
from .errors import register_exception_handlers
from .routers import health as health_router
from .routers import native as native_router
from .routers import openai as openai_router


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger = logging.getLogger("app.lifespan")

    asr = ParakeetASR(settings)
    app.state.asr = asr

    skip_load = os.environ.get("PARAKEET_SKIP_MODEL_LOAD") == "1"
    if skip_load:
        logger.warning(
            "PARAKEET_SKIP_MODEL_LOAD=1 set; skipping model load. "
            "/health will report not-ready and inference endpoints will 503."
        )
    else:
        try:
            asr.load()
            asr.warmup()
        except Exception:
            logger.exception("Failed to load ASR model during startup")
            # Re-raise so the container fails its health check and ECS replaces it.
            raise

    try:
        yield
    finally:
        logger.info("Shutting down")
        app.state.asr = None


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging(settings.log_level)

    app = FastAPI(
        title="Parakeet V3 STT",
        version="0.1.0",
        description=(
            "Stateless FastAPI wrapper around NVIDIA Parakeet TDT 0.6B v3 "
            "multilingual ASR running on CPU via ONNX Runtime."
        ),
        lifespan=lifespan,
    )

    register_exception_handlers(app)

    app.include_router(health_router.router)
    app.include_router(openai_router.router)
    app.include_router(native_router.router)

    return app


# Module-level app for `uvicorn app.main:app`.
app = create_app()
