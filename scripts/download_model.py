#!/usr/bin/env python3
"""Download Parakeet TDT v3 ONNX + Silero VAD into a local directory.

Used both at Docker build time (to bake models into the image) and for local dev.

Usage:
    python scripts/download_model.py [TARGET_DIR] [--quantization int8|fp32|all]

Default TARGET_DIR is ./models (creates ./models/parakeet-v3 and ./models/silero-vad).
Default quantization is int8 (much smaller image; fast on CPU).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
SILERO_REPO = "istupakov/silero-vad-onnx"


def _allow_patterns_for(quantization: str) -> list[str]:
    """Return HF Hub allow_patterns to fetch only what we need."""
    common = ["config.json", "vocab.txt", "nemo128.onnx", "README.md", ".gitattributes"]
    if quantization == "int8":
        return common + ["encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"]
    if quantization == "fp32":
        # Full precision needs the external-data sidecar too.
        return common + [
            "encoder-model.onnx",
            "encoder-model.onnx.data",
            "decoder_joint-model.onnx",
        ]
    if quantization == "all":
        return common + [
            "encoder-model.int8.onnx",
            "decoder_joint-model.int8.onnx",
            "encoder-model.onnx",
            "encoder-model.onnx.data",
            "decoder_joint-model.onnx",
        ]
    raise ValueError(f"Unknown quantization: {quantization!r}")


def download(target_dir: Path, quantization: str = "int8") -> tuple[Path, Path]:
    target_dir = target_dir.resolve()
    parakeet_dir = target_dir / "parakeet-v3"
    silero_dir = target_dir / "silero-vad"
    parakeet_dir.mkdir(parents=True, exist_ok=True)
    silero_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download_model] Fetching {PARAKEET_REPO} ({quantization}) -> {parakeet_dir}", flush=True)
    snapshot_download(
        repo_id=PARAKEET_REPO,
        local_dir=str(parakeet_dir),
        allow_patterns=_allow_patterns_for(quantization),
    )

    print(f"[download_model] Fetching {SILERO_REPO} -> {silero_dir}", flush=True)
    snapshot_download(
        repo_id=SILERO_REPO,
        local_dir=str(silero_dir),
    )

    return parakeet_dir, silero_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target_dir",
        nargs="?",
        default="./models",
        help="Directory to download models into (default: ./models)",
    )
    parser.add_argument(
        "--quantization",
        choices=["int8", "fp32", "all"],
        default="int8",
        help="Which ONNX precision to fetch (default: int8)",
    )
    args = parser.parse_args(argv)

    try:
        parakeet_dir, silero_dir = download(Path(args.target_dir), args.quantization)
    except Exception as exc:  # noqa: BLE001
        print(f"[download_model] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[download_model] Done. parakeet: {parakeet_dir}  silero: {silero_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
