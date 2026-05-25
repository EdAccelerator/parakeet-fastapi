"""Parakeet TDT v3 ASR wrapper.

Loads the ONNX model once (with optional Silero VAD chunking for long audio)
and exposes a thread-safe ``transcribe()`` that runs the actual inference in
the default executor so the FastAPI event loop stays responsive.

Concurrency: a single asyncio.Semaphore (``inference_concurrency``) bounds how
many simultaneous inferences run on the shared ORT session. ORT itself fans
out across cores via ``intra_op_num_threads`` so over-parallelising at the
Python layer just thrashes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    text: str
    duration_s: float
    inference_s: float
    segments: list[Segment] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    language: str | None = None


def _make_session_options(settings: Settings) -> Any:
    """Build ORT SessionOptions tuned for CPU inference."""
    import onnxruntime as ort  # imported lazily so tests can stub

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = settings.ort_intra_op_threads
    opts.inter_op_num_threads = settings.ort_inter_op_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


class ParakeetASR:
    """Thin wrapper around ``onnx_asr`` loaded for Parakeet TDT v3."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: Any = None  # base recognizer (no timestamps)
        self._model_with_vad: Any = None  # for long audio
        self._model_with_timestamps: Any = None  # word/segment timestamps
        self._lock = asyncio.Semaphore(max(1, self._settings.inference_concurrency))
        self._loaded = False

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Synchronously load the ONNX session(s). Call once at startup."""
        if self._loaded:
            return
        import onnx_asr  # imported lazily

        s = self._settings
        model_dir = Path(s.parakeet_model_dir)
        vad_dir = Path(s.parakeet_vad_dir)

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Parakeet model dir not found: {model_dir}. "
                "Run scripts/download_model.py or bake it into the image."
            )

        sess_options = _make_session_options(s)
        providers = ["CPUExecutionProvider"]

        logger.info(
            "Loading Parakeet TDT v3 from %s (quantization=%s, intra_op_threads=%d)",
            model_dir,
            s.parakeet_quantization,
            s.ort_intra_op_threads,
        )
        t0 = time.perf_counter()
        load_kwargs: dict[str, Any] = {
            "path": str(model_dir),
            "sess_options": sess_options,
            "providers": providers,
        }
        if s.quantization_for_loader:
            load_kwargs["quantization"] = s.quantization_for_loader

        base = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", **load_kwargs)
        self._model = base
        self._model_with_timestamps = base.with_timestamps()

        if s.enable_vad:
            if not vad_dir.exists():
                logger.warning(
                    "VAD dir %s not found; disabling VAD chunking. "
                    "Long audio (>20s) may fail.",
                    vad_dir,
                )
            else:
                logger.info("Loading Silero VAD from %s", vad_dir)
                vad = onnx_asr.load_vad(
                    "silero",
                    path=str(vad_dir),
                    sess_options=sess_options,
                    providers=providers,
                )
                self._model_with_vad = base.with_vad(vad)

        elapsed = time.perf_counter() - t0
        logger.info("Parakeet model loaded in %.2fs", elapsed)
        self._loaded = True

    def warmup(self) -> None:
        """Run a tiny dummy inference so the first real request is fast."""
        if not self._loaded:
            return
        try:
            dummy = np.zeros(int(0.5 * self._settings.target_sample_rate), dtype=np.float32)
            # Wrap in list so onnx-asr's batch API is exercised.
            self._model.recognize([dummy], sample_rate=self._settings.target_sample_rate)
            logger.info("ASR warmup complete")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ASR warmup failed (non-fatal): %s", exc)

    # -- inference -----------------------------------------------------------

    async def transcribe(
        self,
        waveform: np.ndarray,
        *,
        sample_rate: int,
        with_timestamps: bool = False,
        use_vad: bool | None = None,
    ) -> TranscriptionResult:
        """Transcribe a single waveform.

        :param waveform: 1-D float32 PCM in [-1, 1].
        :param sample_rate: Sample rate of ``waveform`` (must match the model's
            expected rate; the audio pipeline resamples upstream).
        :param with_timestamps: Return word/segment timestamps.
        :param use_vad: Force VAD on/off. ``None`` → auto: use VAD when the
            waveform is longer than ~20s and VAD is configured.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("ASR model is not loaded")

        duration_s = float(len(waveform)) / float(sample_rate) if sample_rate else 0.0

        if use_vad is None:
            # Auto: VAD for long clips so we don't blow past the model's
            # ~20-30s practical limit. Threshold deliberately conservative.
            use_vad = self._model_with_vad is not None and duration_s > 18.0

        if use_vad and self._model_with_vad is None:
            raise RuntimeError(
                "VAD requested but VAD model is not loaded "
                "(check parakeet_vad_dir and enable_vad)"
            )

        loop = asyncio.get_running_loop()
        async with self._lock:
            t0 = time.perf_counter()
            result = await loop.run_in_executor(
                None,
                self._run_sync,
                waveform,
                sample_rate,
                with_timestamps,
                use_vad,
            )
            inference_s = time.perf_counter() - t0

        result.inference_s = inference_s
        result.duration_s = duration_s
        return result

    def _run_sync(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        with_timestamps: bool,
        use_vad: bool,
    ) -> TranscriptionResult:
        """Blocking inference. Runs in a worker thread."""

        # Each branch returns whatever onnx-asr gives us. We normalise to
        # TranscriptionResult below.
        if use_vad:
            model = self._model_with_vad
            # When chained, the VAD adapter yields segment results.
            # If timestamps were also requested we'd need with_timestamps on
            # the chained adapter; keep it simple: VAD = segments only.
            raw_segments = list(model.recognize(waveform, sample_rate=sample_rate))
            segments: list[Segment] = []
            text_parts: list[str] = []
            for seg in raw_segments:
                seg_text = _extract_text(seg)
                start = float(getattr(seg, "start", 0.0) or 0.0)
                end = float(getattr(seg, "end", 0.0) or 0.0)
                if seg_text:
                    segments.append(Segment(start=start, end=end, text=seg_text))
                    text_parts.append(seg_text)
            return TranscriptionResult(
                text=" ".join(text_parts).strip(),
                duration_s=0.0,  # populated by caller
                inference_s=0.0,
                segments=segments,
            )

        if with_timestamps:
            model = self._model_with_timestamps
            out = model.recognize(waveform, sample_rate=sample_rate)
            text = _extract_text(out)
            words = _extract_words(out)
            segments = _extract_segments(out)
            if not words:
                # onnx-asr returns token-level timestamps; group them into
                # word-level + segment-level structures.
                words = _words_from_tokens(
                    getattr(out, "tokens", None) or [],
                    getattr(out, "timestamps", None) or [],
                )
            if not segments and words:
                segments = _segments_from_words(words, text)
            return TranscriptionResult(
                text=text,
                duration_s=0.0,
                inference_s=0.0,
                segments=segments,
                words=words,
            )

        text = self._model.recognize(waveform, sample_rate=sample_rate)
        if isinstance(text, list):  # batch API safety
            text = text[0] if text else ""
        if not isinstance(text, str):
            text = _extract_text(text)
        return TranscriptionResult(
            text=text,
            duration_s=0.0,
            inference_s=0.0,
        )


# ---------------------------------------------------------------------------
# Normalisation helpers — onnx-asr's result objects vary by adapter chain.
# We probe attributes defensively so a library version bump doesn't break us.
# ---------------------------------------------------------------------------


def _extract_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    for attr in ("text", "transcription"):
        v = getattr(obj, attr, None)
        if isinstance(v, str):
            return v
    # Some adapters expose `.tokens` with stringification.
    return str(obj)


def _extract_words(obj: Any) -> list[Word]:
    raw = getattr(obj, "words", None) or getattr(obj, "word_timestamps", None) or []
    out: list[Word] = []
    for w in raw:
        word = getattr(w, "word", None) or getattr(w, "text", None) or ""
        start = float(getattr(w, "start", 0.0) or 0.0)
        end = float(getattr(w, "end", 0.0) or 0.0)
        if word:
            out.append(Word(word=word, start=start, end=end))
    return out


def _extract_segments(obj: Any) -> list[Segment]:
    raw = getattr(obj, "segments", None) or getattr(obj, "segment_timestamps", None) or []
    out: list[Segment] = []
    for s in raw:
        text = getattr(s, "text", None) or getattr(s, "segment", None) or ""
        start = float(getattr(s, "start", 0.0) or 0.0)
        end = float(getattr(s, "end", 0.0) or 0.0)
        if text:
            out.append(Segment(start=start, end=end, text=text))
    return out


def _words_from_tokens(tokens: list[str], timestamps: list[float]) -> list[Word]:
    """Group BPE-style tokens into words.

    The Parakeet TDT v3 tokenizer emits BPE tokens where word boundaries are
    marked by a leading space (e.g. ``" Hello"``). We accumulate tokens into
    a word until we hit the next space-prefixed token. Each word's start
    timestamp comes from its first token; the end is one position after the
    last token (or extrapolated from the previous step size at the very end).
    """
    if not tokens or not timestamps or len(tokens) != len(timestamps):
        return []

    words: list[Word] = []
    current_tokens: list[str] = []
    current_start: float = 0.0

    def _flush(end_time: float) -> None:
        if not current_tokens:
            return
        joined = "".join(current_tokens).strip()
        if joined:
            words.append(Word(word=joined, start=current_start, end=end_time))

    for i, (tok, ts) in enumerate(zip(tokens, timestamps)):
        # A new word starts when the token begins with a space (or is the very
        # first token of the utterance).
        if tok.startswith(" ") or not current_tokens:
            # Flush the previous word using the current token's start as its end.
            if current_tokens:
                _flush(float(ts))
            current_tokens = [tok]
            current_start = float(ts)
        else:
            current_tokens.append(tok)

    if current_tokens:
        # Estimate the end time for the trailing word.
        if len(timestamps) >= 2:
            step = float(timestamps[-1] - timestamps[-2])
        else:
            step = 0.08
        end_time = float(timestamps[-1]) + max(step, 0.04)
        _flush(end_time)

    return words


def _segments_from_words(words: list[Word], text: str) -> list[Segment]:
    """Roll word-level timestamps up into a single segment spanning all words.

    Parakeet TDT v3 doesn't emit explicit utterance segmentation, so for
    OpenAI-compatible verbose responses we just return one segment with the
    full text and outer-bounds timing.
    """
    if not words:
        return []
    return [
        Segment(
            start=words[0].start,
            end=words[-1].end,
            text=text.strip() or " ".join(w.word for w in words),
        )
    ]
