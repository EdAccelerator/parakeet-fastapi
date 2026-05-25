"""Runtime configuration loaded from environment variables.

All settings are optional with sensible defaults. The service is intended to
run as a stateless container, so configuration is read once at startup.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_intra_op_threads() -> int:
    cpu = os.cpu_count() or 1
    return max(1, cpu)


class Settings(BaseSettings):
    """Service configuration.

    Environment variable names match field names case-insensitively.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # --- Model locations ---
    parakeet_model_dir: str = "/models/parakeet-v3"
    parakeet_vad_dir: str = "/models/silero-vad"
    # int8 is the recommended CPU default; "" / "fp32" / None means non-quantized weights.
    parakeet_quantization: Literal["int8", "fp32"] = "int8"
    parakeet_model_name: str = "parakeet-tdt-0.6b-v3"

    # --- ONNX Runtime ---
    ort_intra_op_threads: int = Field(default_factory=_default_intra_op_threads)
    ort_inter_op_threads: int = 1

    # --- Inference behaviour ---
    inference_concurrency: int = 1  # serial by default; scale via ECS task count
    enable_vad: bool = True  # chunk long audio via Silero VAD

    # --- Limits ---
    max_audio_seconds: float = 1800.0  # 30 minutes
    max_upload_bytes: int = 100 * 1024 * 1024  # 100 MB

    # --- Audio decoding ---
    target_sample_rate: int = 16_000
    ffmpeg_binary: str = "ffmpeg"

    @property
    def quantization_for_loader(self) -> str | None:
        """Map config to the value `onnx_asr.load_model(quantization=...)` expects."""
        return "int8" if self.parakeet_quantization == "int8" else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Tests can override behaviour by clearing the cache: ``get_settings.cache_clear()``.
    """
    return Settings()
