"""Audio decoding tests.

These tests don't load the ASR model — they just exercise the WAV fast path
and the ffmpeg subprocess fallback.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from app.audio import (
    AudioDecodeError,
    AudioTooLongError,
    _resample_linear,
    _to_mono_float32,
    decode_audio,
)
from app.config import Settings

HAS_FFMPEG = shutil.which("ffmpeg") is not None


# ---- helpers ----------------------------------------------------------------


def _settings(**overrides) -> Settings:
    defaults = dict(
        parakeet_model_dir="/tmp/__nope__",
        parakeet_vad_dir="/tmp/__nope__",
        target_sample_rate=16_000,
        max_audio_seconds=60.0,
        max_upload_bytes=10 * 1024 * 1024,
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ---- _to_mono_float32 -------------------------------------------------------


def test_to_mono_float32_already_float32():
    arr = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    out = _to_mono_float32(arr)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, arr)


def test_to_mono_float32_int16_scaling():
    arr = np.array([-32768, 0, 32767], dtype=np.int16)
    out = _to_mono_float32(arr)
    assert out.dtype == np.float32
    assert out[0] == pytest.approx(-1.0)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(32767 / 32768.0, abs=1e-6)


def test_to_mono_float32_downmixes_stereo():
    stereo = np.array([[0.2, 0.4], [0.6, -0.2]], dtype=np.float32)
    out = _to_mono_float32(stereo)
    assert out.ndim == 1
    assert out.shape == (2,)
    np.testing.assert_allclose(out, [0.3, 0.2], atol=1e-6)


# ---- _resample_linear -------------------------------------------------------


def test_resample_linear_noop_when_rates_match():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = _resample_linear(arr, 16000, 16000)
    np.testing.assert_array_equal(out, arr)


def test_resample_linear_doubles_length_when_doubling_rate():
    arr = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    out = _resample_linear(arr, 8000, 16000)
    # Allow off-by-one due to rounding.
    assert abs(out.shape[0] - 2000) <= 1


def test_resample_linear_halves_length_when_halving_rate():
    arr = np.zeros(1000, dtype=np.float32)
    out = _resample_linear(arr, 16000, 8000)
    assert abs(out.shape[0] - 500) <= 1


# ---- decode_audio: WAV fast path --------------------------------------------


def test_decode_wav_via_fast_path(sine_wav_bytes):
    decoded = decode_audio(sine_wav_bytes, _settings())
    assert decoded.sample_rate == 16_000
    assert decoded.waveform.dtype == np.float32
    assert decoded.duration_s == pytest.approx(1.0, abs=0.01)
    # Energy should be non-trivial for a sine wave.
    assert float(np.abs(decoded.waveform).mean()) > 0.1


def test_decode_stereo_wav_is_downmixed_and_resampled(stereo_wav_bytes):
    decoded = decode_audio(stereo_wav_bytes, _settings())
    assert decoded.sample_rate == 16_000
    # 1 s of audio at 16 kHz target → ~16000 samples (resampled from 44.1k)
    assert abs(len(decoded.waveform) - 16_000) < 100
    assert decoded.waveform.ndim == 1


def test_decode_silence_wav_returns_zeros(silence_wav_bytes):
    decoded = decode_audio(silence_wav_bytes, _settings())
    assert decoded.duration_s == pytest.approx(0.5, abs=0.01)
    assert float(np.max(np.abs(decoded.waveform))) == 0.0


# ---- decode_audio: error handling -------------------------------------------


def test_decode_empty_bytes_raises():
    with pytest.raises(AudioDecodeError):
        decode_audio(b"", _settings())


def test_decode_garbage_bytes_raises():
    # Random bytes that aren't a valid WAV/FLAC and ffmpeg also can't decode.
    with pytest.raises(AudioDecodeError):
        decode_audio(b"this is clearly not audio data", _settings())


def test_decode_rejects_audio_longer_than_max_seconds(long_sine_wav_bytes):
    # 5 s wav, cap at 1 s
    with pytest.raises(AudioTooLongError):
        decode_audio(long_sine_wav_bytes, _settings(max_audio_seconds=1.0))


def test_decode_uses_ffmpeg_when_fast_path_unavailable(sine_wav_bytes, monkeypatch):
    """Force the ffmpeg fallback by making the soundfile path return None."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")

    import app.audio as audio_mod

    monkeypatch.setattr(audio_mod, "_decode_via_soundfile", lambda raw, sr: None)
    decoded = audio_mod.decode_audio(sine_wav_bytes, _settings())
    assert decoded.sample_rate == 16_000
    assert decoded.duration_s == pytest.approx(1.0, abs=0.05)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_decode_mp3_via_ffmpeg(sine_wav_bytes):
    """End-to-end: convert WAV to MP3 with ffmpeg, then decode it back."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "5",
            "-f",
            "mp3",
            "pipe:1",
        ],
        input=sine_wav_bytes,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        pytest.skip("libmp3lame not available in local ffmpeg build")

    decoded = decode_audio(proc.stdout, _settings(), filename_hint="x.mp3")
    assert decoded.sample_rate == 16_000
    assert decoded.duration_s == pytest.approx(1.0, abs=0.1)


def test_decode_ffmpeg_missing_raises_clear_error(sine_wav_bytes, monkeypatch):
    """If ffmpeg isn't on PATH AND the fast path is bypassed, we get a clear error."""
    import app.audio as audio_mod

    monkeypatch.setattr(audio_mod, "_decode_via_soundfile", lambda raw, sr: None)
    monkeypatch.setattr(audio_mod.shutil, "which", lambda _: None)
    with pytest.raises(AudioDecodeError, match="ffmpeg"):
        audio_mod.decode_audio(sine_wav_bytes, _settings())
