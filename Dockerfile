# syntax=docker/dockerfile:1.6

# ---------- Stage 1: fetch ONNX models from Hugging Face ----------
FROM python:3.12-slim AS model-fetch

ARG PARAKEET_QUANTIZATION=int8
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir "huggingface-hub>=0.24,<1.0"

WORKDIR /build
COPY scripts/download_model.py /build/scripts/download_model.py

# Use HF_HUB_ENABLE_HF_TRANSFER if available; harmless if not.
ENV HF_HUB_ENABLE_HF_TRANSFER=0

RUN python /build/scripts/download_model.py /models --quantization "${PARAKEET_QUANTIZATION}"

# ---------- Stage 2: runtime image ----------
FROM python:3.12-slim AS runtime

# System deps: ffmpeg for audio decoding, libsndfile1 for soundfile fast path.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PARAKEET_MODEL_DIR=/models/parakeet-v3 \
    PARAKEET_VAD_DIR=/models/silero-vad \
    PARAKEET_QUANTIZATION=int8 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Install Python deps first for cache-friendly layering.
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir .

# Copy application code.
COPY app /app/app
COPY scripts /app/scripts

# Copy baked-in models from the fetch stage.
COPY --from=model-fetch /models /models

# Create non-root user.
RUN useradd --create-home --shell /bin/bash --uid 10001 parakeet \
 && chown -R parakeet:parakeet /app /models
USER parakeet

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST} --port ${PORT} --workers 1"]
