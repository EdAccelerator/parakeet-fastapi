"""Real-model integration tests.

These are marked `slow` and skipped by default. To run them:

    PARAKEET_MODEL_DIR=/path/to/models/parakeet-v3 \
    PARAKEET_VAD_DIR=/path/to/models/silero-vad \
    pytest tests/integration -m slow

If the model isn't present and ``ALLOW_MODEL_DOWNLOAD=1`` is set in the env,
the test will download the int8 weights (~700 MB) into ``./_tmp_models``.

The sample WAV is the NVIDIA-recommended LibriSpeech clip from the Parakeet
model card. It's a passage from Nathaniel Hawthorne's "The House of the
Seven Gables" (speaker 2086, chapter 149220, utterance 33).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import pytest

# Same WAV NVIDIA links from the Parakeet model card.
SAMPLE_URL = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
# Words known to appear in the transcription (used for loose verification).
EXPECTED_TOKENS = ("phoebe", "portrait", "observed")

pytestmark = [pytest.mark.slow]


def _ensure_sample(target: Path) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[integration] downloading sample audio -> {target}")
    urllib.request.urlretrieve(SAMPLE_URL, target)
    return target


def _ensure_models(root: Path) -> tuple[Path, Path]:
    parakeet_dir = root / "parakeet-v3"
    silero_dir = root / "silero-vad"
    needs_download = not (parakeet_dir.exists() and any(parakeet_dir.iterdir()))
    if needs_download:
        if os.environ.get("ALLOW_MODEL_DOWNLOAD") != "1":
            pytest.skip(
                "Models not present and ALLOW_MODEL_DOWNLOAD!=1. "
                "Set ALLOW_MODEL_DOWNLOAD=1 to fetch ~700 MB."
            )
        from scripts.download_model import download

        download(root, quantization="int8")
    return parakeet_dir, silero_dir


@pytest.fixture(scope="session")
def real_asr():
    """Load the real Parakeet ASR once per test session."""
    from app.asr import ParakeetASR
    from app.config import Settings

    model_dir = os.environ.get("PARAKEET_MODEL_DIR")
    vad_dir = os.environ.get("PARAKEET_VAD_DIR")
    if not (model_dir and Path(model_dir).exists()):
        tmp_root = Path(os.environ.get("MODEL_CACHE", "./_tmp_models")).resolve()
        parakeet_dir, silero_dir = _ensure_models(tmp_root)
        model_dir = str(parakeet_dir)
        vad_dir = str(silero_dir)

    settings = Settings(
        parakeet_model_dir=model_dir,
        parakeet_vad_dir=vad_dir,
        parakeet_quantization="int8",
    )
    asr = ParakeetASR(settings)
    asr.load()
    return asr


@pytest.mark.asyncio
async def test_transcribes_known_audio(real_asr, tmp_path):
    """End-to-end: decode + transcribe a known speech clip."""
    from app.audio import decode_audio
    from app.config import Settings

    wav_path = tmp_path / "sample.wav"
    _ensure_sample(wav_path)

    settings = Settings(
        parakeet_model_dir="/dev/null",  # unused — decode only
        parakeet_vad_dir="/dev/null",
    )
    decoded = decode_audio(wav_path.read_bytes(), settings)
    result = await real_asr.transcribe(decoded.waveform, sample_rate=decoded.sample_rate)

    text = result.text.lower()
    print(f"[integration] transcription: {result.text!r} ({result.inference_s:.2f}s)")
    assert len(text) > 5
    # Loose check: at least one expected token must appear. Parakeet is good
    # but exact matches can drift across model versions/quantization.
    assert any(token in text for token in EXPECTED_TOKENS), (
        f"None of {EXPECTED_TOKENS} found in transcription: {result.text!r}"
    )


@pytest.mark.asyncio
async def test_transcribe_silence_returns_empty_or_short(real_asr):
    silence = np.zeros(16_000, dtype=np.float32)
    result = await real_asr.transcribe(silence, sample_rate=16_000)
    # Silence should produce very short or empty output.
    assert len(result.text.strip()) < 20


@pytest.mark.asyncio
async def test_transcribe_with_timestamps(real_asr, tmp_path):
    from app.audio import decode_audio
    from app.config import Settings

    wav_path = tmp_path / "sample.wav"
    _ensure_sample(wav_path)
    decoded = decode_audio(
        wav_path.read_bytes(),
        Settings(parakeet_model_dir="/dev/null", parakeet_vad_dir="/dev/null"),
    )
    result = await real_asr.transcribe(
        decoded.waveform,
        sample_rate=decoded.sample_rate,
        with_timestamps=True,
    )
    assert result.text.strip()
    print(f"[integration] {len(result.words)} words, {len(result.segments)} segments")
    assert result.words, "Expected non-empty word timestamps"
    assert result.segments, "Expected at least one segment"
    # Timestamps must be monotonically non-decreasing and within audio bounds.
    duration = result.duration_s
    prev_end = 0.0
    for w in result.words:
        assert 0.0 <= w.start <= w.end
        assert w.end <= duration + 1.0
        assert w.start + 1e-6 >= prev_end - 0.5  # allow tiny overlaps
        prev_end = w.end
