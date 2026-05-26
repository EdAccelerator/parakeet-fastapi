# parakeet-fastapi

A stateless, Dockerized FastAPI wrapper around NVIDIA's
[Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
multilingual speech-to-text model. Inference runs on **CPU** via
[ONNX Runtime](https://onnxruntime.ai/), backed by the
[`onnx-asr`](https://github.com/istupakov/onnx-asr) library and
[`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
(int8 quantized by default — ~670 MB, ~3-4× faster than fp32 on CPU).

- 25-language multilingual ASR with punctuation and capitalization
- Drop-in **OpenAI-compatible** `/v1/audio/transcriptions` endpoint
- Richer native `/transcribe` endpoint with timestamps and VAD chunking
- Silero VAD auto-chunks audio longer than ~20 s
- Decodes WAV / FLAC / MP3 / M4A / OGG / WebM / Opus etc. via bundled `ffmpeg`
- Stateless — no disk writes, no shared state. Scales horizontally on
  AWS ECS / Fargate
- Configurable thread count + concurrency for tuning to your CPU shape

## Quickstart (Docker)

Pull the pre-built image from GitHub Container Registry:

```sh
docker run --rm -p 8000:8000 ghcr.io/edaccelerator/parakeet-fastapi:latest
```

Or build it yourself:

```sh
docker build -t parakeet-fastapi:latest .
docker run --rm -p 8000:8000 parakeet-fastapi:latest
```

Wait ~30 s for the model to load, then:

```sh
curl -fsS http://localhost:8000/health
# {"status":"ok","model_loaded":true,"model_name":"parakeet-tdt-0.6b-v3", ...}

curl -fsS -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=parakeet-tdt-0.6b-v3"
# {"text":"..."}
```

OpenAI Python SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
with open("sample.wav", "rb") as f:
    resp = client.audio.transcriptions.create(model="parakeet-tdt-0.6b-v3", file=f)
print(resp.text)
```

## API

| Method | Path                          | Description                                |
| ------ | ----------------------------- | ------------------------------------------ |
| GET    | `/`                           | Service info                               |
| GET    | `/health`                     | Readiness probe (503 until model loaded)   |
| GET    | `/v1/models`                  | OpenAI-style model list                    |
| POST   | `/v1/audio/transcriptions`    | OpenAI-compatible transcription            |
| POST   | `/transcribe`                 | Native transcription w/ timestamps + VAD   |
| GET    | `/docs`                       | OpenAPI interactive docs                   |

### `POST /v1/audio/transcriptions`

`multipart/form-data` fields (matching the OpenAI Audio API where possible):

- `file` *(required)* — audio bytes
- `model` *(optional)* — informational; only one model is loaded
- `language` *(optional)* — informational; Parakeet v3 auto-detects
- `prompt` *(optional)* — accepted for compatibility, ignored
- `temperature` *(optional)* — accepted, ignored (TDT decoding is greedy)
- `response_format` *(optional)* — `json` (default) | `text` | `verbose_json` | `srt` | `vtt`
- `timestamp_granularities` *(optional)* — implies timestamps

### `POST /transcribe`

`multipart/form-data` fields:

- `file` *(required)*
- `timestamps` *(bool, default false)* — return word + segment timestamps
- `vad` *(bool, default auto)* — force VAD chunking on/off; auto enables it for clips > ~18 s
- `language` *(optional)* — informational
- `response_format` — same options as the OpenAI endpoint

## Configuration

All settings come from environment variables. Defaults shown:

| Variable                  | Default                  | Notes                                                          |
| ------------------------- | ------------------------ | -------------------------------------------------------------- |
| `HOST`                    | `0.0.0.0`                |                                                                |
| `PORT`                    | `8000`                   |                                                                |
| `LOG_LEVEL`               | `INFO`                   | `DEBUG`/`INFO`/`WARNING`/`ERROR`                               |
| `PARAKEET_MODEL_DIR`      | `/models/parakeet-v3`    | Where the baked-in ONNX files live                             |
| `PARAKEET_VAD_DIR`        | `/models/silero-vad`     |                                                                |
| `PARAKEET_QUANTIZATION`   | `int8`                   | `int8` or `fp32`                                               |
| `ORT_INTRA_OP_THREADS`    | *(CPU count)*            | ONNX intra-op threads                                          |
| `ORT_INTER_OP_THREADS`    | `1`                      |                                                                |
| `INFERENCE_CONCURRENCY`   | `1`                      | In-process concurrency cap; scale via ECS task count instead   |
| `ENABLE_VAD`              | `true`                   | Disable to skip Silero VAD entirely                            |
| `MAX_AUDIO_SECONDS`       | `1800` (30 min)          | Reject longer inputs with 413                                  |
| `MAX_UPLOAD_BYTES`        | `104857600` (100 MB)     |                                                                |
| `TARGET_SAMPLE_RATE`      | `16000`                  | Parakeet's native rate; don't change                           |
| `FFMPEG_BINARY`           | `ffmpeg`                 |                                                                |

## Run locally with Docker (recommended)

```sh
git clone git@github.com:EdAccelerator/parakeet-fastapi.git
cd parakeet-fastapi

docker build -t parakeet-fastapi:latest .
docker run --rm -p 8000:8000 parakeet-fastapi:latest
```

The first build takes ~3-5 minutes (downloading the ~670 MB int8 ONNX model
during stage 1) and produces a ~3.4 GB image. Subsequent builds reuse the
cached model layer.

Or with Compose:

```sh
docker compose up --build
```

Once running, smoke test it:

```sh
curl -fsS http://localhost:8000/health

curl -sS -o sample.wav \
  https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav

curl -fsS -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=parakeet-tdt-0.6b-v3"
# {"text":"Well, I don't wish to see it any more, observed Phoebe..."}
```

## Run locally without Docker

Useful for fast iteration with `--reload`.

```sh
# 1. System deps (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 python3.12 python3.12-venv

# 2. Create a venv and install the package + dev deps
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# 3. Download models into ./models (~670 MB int8 + ~6 MB Silero VAD)
python scripts/download_model.py ./models --quantization int8

# 4. Run the server
PARAKEET_MODEL_DIR=$PWD/models/parakeet-v3 \
PARAKEET_VAD_DIR=$PWD/models/silero-vad \
uvicorn app.main:app --reload --port 8000
```

The server takes ~5-10 s to load the model on a typical laptop. Then visit
[http://localhost:8000/docs](http://localhost:8000/docs) for the interactive
OpenAPI UI.

> If your distro doesn't ship Python 3.12, install [`uv`](https://docs.astral.sh/uv/)
> and let it manage the interpreter:
> ```sh
> curl -LsSf https://astral.sh/uv/install.sh | sh
> uv python install 3.12
> uv venv --python 3.12 .venv
> source .venv/bin/activate
> uv pip install -e ".[dev]"
> ```

## Tests

All three tiers can be run independently. From the repo root with the venv
activated:

### Unit tests — fast, no model, no network (~1 s)

```sh
pytest tests/unit -q
```

Covers config parsing, schemas, audio decoding (WAV fast path + ffmpeg
fallback), token-to-word grouping, and SRT/VTT formatting.

### Integration tests — endpoint contract + real model

Endpoint contract tests use a fake ASR backend and are fast:

```sh
pytest tests/integration/test_endpoints.py -q
```

Real-model integration tests are marked `slow` and skipped by default. They
need the ONNX model on disk:

```sh
# Once: download models if you haven't already.
python scripts/download_model.py ./models --quantization int8

PARAKEET_MODEL_DIR=$PWD/models/parakeet-v3 \
PARAKEET_VAD_DIR=$PWD/models/silero-vad \
pytest tests/integration -m slow -q
```

If you'd rather the test suite download the model itself:

```sh
ALLOW_MODEL_DOWNLOAD=1 pytest tests/integration -m slow -q
```

### End-to-end Docker tests — builds the image and hits a live container

```sh
# Build + run + verify (slow: builds image if not cached, ~3-5 min first time)
pytest tests/e2e -m docker -q

# Or skip the build and reuse an existing tag:
docker build -t parakeet-fastapi:e2e .
PARAKEET_IMAGE=parakeet-fastapi:e2e PARAKEET_SKIP_BUILD=1 \
  pytest tests/e2e -m docker -q
```

### Everything except the Docker layer

```sh
PARAKEET_MODEL_DIR=$PWD/models/parakeet-v3 \
PARAKEET_VAD_DIR=$PWD/models/silero-vad \
pytest -m "not docker" -q
```

## Container image

Published to GitHub Container Registry by the
[`Publish container image`](.github/workflows/publish-image.yml) workflow on
every push to `main` and on version tags. Built as a multi-arch manifest
for both `linux/amd64` (ECS Fargate, most cloud VMs, classic CI) and
`linux/arm64` (AWS Graviton, Apple Silicon developer laptops, Raspberry
Pi 4/5). Each arch is built on its own native GitHub-hosted runner —
`ubuntu-latest` for amd64 and `ubuntu-24.04-arm` for arm64 — so neither
build pays the QEMU emulation tax. The two single-arch images are
stitched into a manifest list at the end of the workflow, so a single
pull resolves to the right binary automatically.

```
ghcr.io/edaccelerator/parakeet-fastapi:latest          # tip of main
ghcr.io/edaccelerator/parakeet-fastapi:main            # same
ghcr.io/edaccelerator/parakeet-fastapi:sha-<short-sha> # exact commit
ghcr.io/edaccelerator/parakeet-fastapi:v0.1.0          # version tag (when cut)
ghcr.io/edaccelerator/parakeet-fastapi:0.1             # major.minor
```

The image is **public** once you flip the package visibility to public the
first time (`Repo -> Packages -> parakeet-fastapi -> Package settings ->
Change visibility -> Public`). Until then, pulls require a personal access
token with `read:packages`:

```sh
echo "$GHCR_PAT" | docker login ghcr.io -u <your-user> --password-stdin
docker pull ghcr.io/edaccelerator/parakeet-fastapi:latest
```

Each pushed image carries an
[SLSA-3 provenance attestation](https://github.com/EdAccelerator/parakeet-fastapi/attestations)
produced via `actions/attest-build-provenance`. Verify with:

```sh
gh attestation verify oci://ghcr.io/edaccelerator/parakeet-fastapi:latest \
  --owner EdAccelerator
```

## AWS ECS deployment

The image is designed for **AWS ECS (Fargate or EC2)**:

- **Recommended task size:** 2 vCPU / 4 GB RAM minimum (int8). Bumping
  to 4 vCPU roughly halves single-request latency since ORT scales
  intra-op threads with cores.
- **Cold start:** ~10–25 s on Fargate (~1.4 GB image pull + model load).
- **Scaling:** keep `INFERENCE_CONCURRENCY=1` and scale horizontally on
  request count or CPU utilization. The model is CPU-bound and a single
  ORT session already saturates the available cores.
- **Health check:** point the ALB/target-group health check at `/health`
  with a `start_period` of ≥ 90 s.
- **Architecture:** the published image is multi-arch (linux/amd64 +
  linux/arm64), so the same tag works on standard Fargate (amd64) and
  Fargate on AWS Graviton (arm64). Graviton is generally ~20% cheaper
  per vCPU-hour at comparable performance for ONNX int8 inference; set
  `"runtimePlatform": {"cpuArchitecture": "ARM64", "operatingSystemFamily":
  "LINUX"}` in the task definition to opt in.

Minimal task definition snippet:

```json
{
  "family": "parakeet-fastapi",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [{
    "name": "parakeet",
    "image": "ghcr.io/edaccelerator/parakeet-fastapi:latest",
    "portMappings": [{ "containerPort": 8000, "protocol": "tcp" }],
    "environment": [
      { "name": "LOG_LEVEL", "value": "INFO" },
      { "name": "INFERENCE_CONCURRENCY", "value": "1" }
    ],
    "healthCheck": {
      "command": ["CMD-SHELL", "curl -fsS http://127.0.0.1:8000/health || exit 1"],
      "interval": 30, "timeout": 5, "retries": 5, "startPeriod": 120
    },
    "essential": true
  }]
}
```

## How it works (and what we deliberately didn't do)

- We use `onnx-asr` so the container ships only what's needed for inference:
  numpy + onnxruntime + a tiny tokenizer/decoder. No PyTorch, no NeMo, no
  CUDA libs in the image.
- The model is **baked into the image at build time** (Dockerfile stage 1),
  so cold starts don't require network access — important on ECS.
- We default to **int8** quantization. The accuracy hit vs fp32 is small
  (a few tenths of a percent WER on the open ASR leaderboard) but image
  size shrinks from ~3.5 GB to ~1.4 GB and CPU inference is ~3-4× faster.
- Long audio (>~20 s) is automatically split via **Silero VAD** ONNX. The
  base Parakeet model can only attend to ~20-30 s at once on CPU; for
  multi-minute audio you need chunking.
- No streaming / websocket. That's a separate feature scope; long-form
  batch transcription is handled via VAD on a single request.
- No multi-worker uvicorn. The 700 MB ONNX session would be duplicated per
  worker. ECS horizontal scaling is the right knob.

## License

Apache-2.0 for this wrapper code. The Parakeet model weights are CC-BY-4.0
(NVIDIA); see the [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
