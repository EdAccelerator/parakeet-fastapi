"""End-to-end Docker container test.

Builds the image (if not already present), starts a container, polls
``/health`` until ready, then exercises the OpenAI-compatible endpoint with
a real audio file. The test is marked ``docker`` and skipped by default.

To run:

    pytest tests/e2e -m docker -q

You can skip the (slow) image build and reuse a pre-built one by setting:

    PARAKEET_IMAGE=parakeet-fastapi:test pytest tests/e2e -m docker -q
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest
import requests

pytestmark = [pytest.mark.docker]

HAS_DOCKER = shutil.which("docker") is not None
IMAGE = os.environ.get("PARAKEET_IMAGE", "parakeet-fastapi:e2e")
# LibriSpeech sample linked from the Parakeet model card.
SAMPLE_URL = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
EXPECTED_TOKENS = ("phoebe", "portrait", "observed")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _docker_available() -> bool:
    if not HAS_DOCKER:
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _image_exists(tag: str) -> bool:
    proc = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def _build_image(tag: str) -> None:
    print(f"[e2e] building {tag} ... (this can take 10+ minutes the first time)")
    proc = subprocess.run(
        ["docker", "build", "-t", tag, "."],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker build failed with rc={proc.returncode}")


@pytest.fixture(scope="session")
def container():
    if not _docker_available():
        pytest.skip("Docker is not available in this environment")

    if not _image_exists(IMAGE):
        if os.environ.get("PARAKEET_SKIP_BUILD") == "1":
            pytest.skip(f"Image {IMAGE} not present and PARAKEET_SKIP_BUILD=1 set")
        _build_image(IMAGE)

    port = _free_port()
    name = f"parakeet-e2e-{port}"

    # Clean any stale container with the same name.
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)

    print(f"[e2e] starting {name} on host port {port}")
    proc = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-p",
            f"{port}:8000",
            "-e",
            "LOG_LEVEL=INFO",
            IMAGE,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker run failed: {proc.stderr}")

    base = f"http://127.0.0.1:{port}"
    # Wait for /health to report ready. Allow up to 5 minutes for cold load.
    deadline = time.time() + 300
    last_err: str | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base}/health", timeout=5)
            if r.status_code == 200 and r.json().get("model_loaded"):
                break
            last_err = f"status={r.status_code} body={r.text[:200]}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(2)
    else:
        # Dump logs to help debug, then fail.
        logs = subprocess.run(
            ["docker", "logs", name], capture_output=True, text=True, check=False
        )
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
        raise RuntimeError(
            f"Container did not become healthy in time. Last error: {last_err}\n"
            f"--- container logs ---\n{logs.stdout}\n{logs.stderr}"
        )

    yield base

    print(f"[e2e] stopping {name}")
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def test_root_endpoint(container):
    r = requests.get(f"{container}/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "parakeet-fastapi"


def test_health_endpoint(container):
    r = requests.get(f"{container}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_models_endpoint(container):
    r = requests.get(f"{container}/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["data"][0]["id"]


def test_openai_transcription_with_real_audio(container, tmp_path):
    wav = tmp_path / "sample.wav"
    if not wav.exists():
        urllib.request.urlretrieve(SAMPLE_URL, wav)

    with wav.open("rb") as f:
        r = requests.post(
            f"{container}/v1/audio/transcriptions",
            files={"file": ("sample.wav", f, "audio/wav")},
            data={"model": "parakeet-tdt-0.6b-v3"},
            timeout=120,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    text = body["text"].lower()
    print(f"[e2e] transcription: {body['text']!r}")
    assert len(text) > 5
    assert any(tok in text for tok in EXPECTED_TOKENS), (
        f"None of {EXPECTED_TOKENS} found in: {body['text']!r}"
    )


def test_native_transcribe_with_timestamps(container, tmp_path):
    wav = tmp_path / "sample.wav"
    if not wav.exists():
        urllib.request.urlretrieve(SAMPLE_URL, wav)
    with wav.open("rb") as f:
        r = requests.post(
            f"{container}/transcribe",
            files={"file": ("sample.wav", f, "audio/wav")},
            data={"timestamps": "true"},
            timeout=120,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"]
    assert body["duration"] > 0


def test_rejects_bad_audio(container):
    r = requests.post(
        f"{container}/v1/audio/transcriptions",
        files={"file": ("garbage.bin", b"not audio at all", "application/octet-stream")},
    )
    assert r.status_code == 400
