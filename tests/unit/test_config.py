"""Settings parsing tests."""

from __future__ import annotations

import os
from unittest import mock

from app.config import Settings, get_settings


def test_defaults_have_reasonable_values():
    s = Settings()
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.parakeet_quantization == "int8"
    assert s.target_sample_rate == 16_000
    assert s.max_audio_seconds == 1800.0
    assert s.inference_concurrency >= 1
    assert s.ort_intra_op_threads >= 1


def test_env_overrides_settings():
    with mock.patch.dict(
        os.environ,
        {
            "PORT": "9001",
            "LOG_LEVEL": "DEBUG",
            "PARAKEET_QUANTIZATION": "fp32",
            "MAX_AUDIO_SECONDS": "60",
            "ORT_INTRA_OP_THREADS": "7",
        },
        clear=False,
    ):
        s = Settings()
        assert s.port == 9001
        assert s.log_level == "DEBUG"
        assert s.parakeet_quantization == "fp32"
        assert s.max_audio_seconds == 60.0
        assert s.ort_intra_op_threads == 7


def test_get_settings_is_cached():
    a = get_settings()
    b = get_settings()
    assert a is b


def test_quantization_for_loader_mapping():
    s_int8 = Settings(parakeet_quantization="int8")
    s_fp32 = Settings(parakeet_quantization="fp32")
    assert s_int8.quantization_for_loader == "int8"
    assert s_fp32.quantization_for_loader is None
