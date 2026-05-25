"""Audio decoding for STT input.

Two paths:

1. ``decode_audio()`` (the public entry point) accepts arbitrary audio bytes and
   returns ``(waveform: np.ndarray[float32], sample_rate: int)`` resampled to
   the target sample rate, mono.
2. The implementation first tries a ``soundfile`` fast path for WAV/FLAC, then
   falls back to a streamed ``ffmpeg`` subprocess for everything else
   (mp3/m4a/ogg/webm/opus/...). ffmpeg is the standard portable decoder.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

from .config import Settings, get_settings


class AudioDecodeError(ValueError):
    """Raised when audio bytes cannot be decoded by either path."""


class AudioTooLongError(ValueError):
    """Raised when decoded audio exceeds ``max_audio_seconds``."""


@dataclass(frozen=True)
class DecodedAudio:
    waveform: np.ndarray  # 1-D float32, range [-1, 1]
    sample_rate: int

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.waveform)) / float(self.sample_rate)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_mono_float32(arr: np.ndarray) -> np.ndarray:
    """Convert any shape/dtype ndarray to 1-D float32 in [-1, 1]."""
    if arr.ndim > 1:
        # soundfile returns (frames, channels) for multichannel.
        arr = arr.mean(axis=1)

    if arr.dtype == np.float32:
        return np.ascontiguousarray(arr)
    if arr.dtype == np.float64:
        return arr.astype(np.float32, copy=False)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        # Scale to [-1, 1]. For signed types the max abs is |min|.
        max_abs = float(max(abs(info.min), info.max))
        return (arr.astype(np.float32) / max_abs).astype(np.float32, copy=False)
    # Unknown numeric type — try cast.
    return arr.astype(np.float32, copy=False)


def _resample_linear(waveform: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Cheap linear resampler. Good enough as a fallback; ffmpeg handles the
    common case with proper anti-aliasing in the slow path."""
    if src_sr == dst_sr or waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    duration = waveform.shape[0] / float(src_sr)
    new_len = int(round(duration * dst_sr))
    if new_len <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, duration, num=waveform.shape[0], endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, duration, num=new_len, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, waveform).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Fast path: soundfile (WAV/FLAC, OGG-Vorbis if libsndfile built with it)
# ---------------------------------------------------------------------------


def _decode_via_soundfile(raw: bytes, target_sr: int) -> DecodedAudio | None:
    try:
        import soundfile as sf  # type: ignore
    except Exception:  # pragma: no cover - soundfile is a hard dep
        return None
    try:
        with sf.SoundFile(io.BytesIO(raw)) as f:
            data = f.read(dtype="float32", always_2d=False)
            src_sr = int(f.samplerate)
    except Exception:
        return None

    wave = _to_mono_float32(np.asarray(data))
    if src_sr != target_sr:
        wave = _resample_linear(wave, src_sr, target_sr)
    return DecodedAudio(waveform=wave, sample_rate=target_sr)


# ---------------------------------------------------------------------------
# Fallback path: ffmpeg subprocess (universal decoder)
# ---------------------------------------------------------------------------


def _decode_via_ffmpeg(
    raw: bytes,
    target_sr: int,
    ffmpeg_binary: str,
) -> DecodedAudio:
    """Pipe bytes through ``ffmpeg`` and read mono s16le PCM at ``target_sr``."""
    if not shutil.which(ffmpeg_binary):
        raise AudioDecodeError(
            f"ffmpeg binary {ffmpeg_binary!r} not found on PATH; "
            "install ffmpeg or send raw WAV/FLAC."
        )

    cmd = [
        ffmpeg_binary,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=raw,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AudioDecodeError(f"ffmpeg not found: {exc}") from exc
    except OSError as exc:  # pragma: no cover - unusual environment failure
        raise AudioDecodeError(f"ffmpeg invocation failed: {exc}") from exc

    if proc.returncode != 0 or not proc.stdout:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(
            f"ffmpeg failed to decode audio (rc={proc.returncode}): "
            f"{stderr[:500] or '<no stderr>'}"
        )

    samples_i16 = np.frombuffer(proc.stdout, dtype=np.int16)
    waveform = samples_i16.astype(np.float32) / 32768.0
    return DecodedAudio(waveform=waveform, sample_rate=target_sr)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def decode_audio(
    raw: bytes,
    settings: Settings | None = None,
    *,
    filename_hint: str | None = None,
) -> DecodedAudio:
    """Decode arbitrary audio bytes to mono float32 PCM at ``target_sample_rate``.

    Strategy:

    - If the byte stream sniffs as a RIFF/WAV or FLAC, try ``soundfile`` first.
    - Otherwise (or on soundfile failure) hand off to ffmpeg.

    Raises :class:`AudioDecodeError` for unrecognised / corrupt input and
    :class:`AudioTooLongError` if duration exceeds the configured cap.
    """
    if not raw:
        raise AudioDecodeError("Empty audio payload")

    settings = settings or get_settings()
    target_sr = settings.target_sample_rate

    looks_like_wav = raw[:4] == b"RIFF" or raw[:4] == b"FORM"
    looks_like_flac = raw[:4] == b"fLaC"
    hint_ext = (filename_hint or "").lower().rsplit(".", 1)[-1] if filename_hint else ""

    decoded: DecodedAudio | None = None
    if looks_like_wav or looks_like_flac or hint_ext in {"wav", "flac"}:
        decoded = _decode_via_soundfile(raw, target_sr)

    if decoded is None:
        decoded = _decode_via_ffmpeg(raw, target_sr, settings.ffmpeg_binary)

    if decoded.waveform.size == 0:
        raise AudioDecodeError("Decoded audio is empty")

    if decoded.duration_s > settings.max_audio_seconds:
        raise AudioTooLongError(
            f"Audio is {decoded.duration_s:.1f}s, exceeds "
            f"max_audio_seconds={settings.max_audio_seconds:.1f}s"
        )
    return decoded
