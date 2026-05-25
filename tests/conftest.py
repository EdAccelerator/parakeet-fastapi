"""Shared pytest fixtures.

We generate a tiny synthetic WAV fixture programmatically so the repository
doesn't need to ship binary audio. The integration tests use real audio (the
JFK clip) that they download or generate on demand — see tests/integration.
"""

from __future__ import annotations

import io
import os
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

# Ensure the repo root is on sys.path so `import app` works without installation.
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _write_wav(samples_int16: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.astype("<i2").tobytes())
    return buf.getvalue()


@pytest.fixture(scope="session")
def sine_wav_bytes() -> bytes:
    """A 1-second 440 Hz sine wave at 16 kHz, signed 16-bit PCM mono WAV."""
    sr = 16_000
    duration = 1.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    amp = 0.5 * 32_767
    samples = (amp * np.sin(2 * np.pi * 440.0 * t)).astype(np.int16)
    return _write_wav(samples, sr)


@pytest.fixture(scope="session")
def silence_wav_bytes() -> bytes:
    """0.5 s of silence at 16 kHz, mono."""
    sr = 16_000
    return _write_wav(np.zeros(int(sr * 0.5), dtype=np.int16), sr)


@pytest.fixture(scope="session")
def long_sine_wav_bytes() -> bytes:
    """5 s of 220 Hz sine at 16 kHz — long enough to be non-trivial but short."""
    sr = 16_000
    duration = 5.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    samples = (0.3 * 32_767 * np.sin(2 * np.pi * 220.0 * t)).astype(np.int16)
    return _write_wav(samples, sr)


@pytest.fixture(scope="session")
def stereo_wav_bytes() -> bytes:
    """1 s of stereo audio (different tone per channel) at 44.1 kHz, 16-bit."""
    sr = 44_100
    duration = 1.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    left = (0.4 * 32_767 * np.sin(2 * np.pi * 440.0 * t)).astype(np.int16)
    right = (0.4 * 32_767 * np.sin(2 * np.pi * 880.0 * t)).astype(np.int16)
    interleaved = np.empty(left.size * 2, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(interleaved.tobytes())
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Each test gets a fresh Settings instance so env overrides take effect."""
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
