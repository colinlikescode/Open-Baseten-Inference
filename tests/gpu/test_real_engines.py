"""Real NVIDIA GPU smoke tests (``pytest -m gpu tests/gpu``).

They run only when NVML sees at least one GPU and vLLM or SGLang is installed (in this
environment or via ``SERVEPILOT_VLLM_PYTHON`` / ``SERVEPILOT_SGLANG_PYTHON``). A very small public
model keeps download and startup times reasonable. Never part of the default CI run.
"""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from servepilot.engines.registry import build_registry
from servepilot.runtime.ports import ephemeral_port
from servepilot.settings import ServePilotSettings

pytestmark = pytest.mark.gpu

SMOKE_MODEL = os.environ.get("SERVEPILOT_GPU_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def _gpu_available() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        from servepilot.hardware.nvml import NVMLHardwareProvider

        return NVMLHardwareProvider().snapshot().gpu_count > 0
    except Exception:
        return False


def _available_engines() -> list[str]:
    return [
        e.name() for e in build_registry(ServePilotSettings()).available() if e.name() != "fake"
    ]


HAS_GPU = _gpu_available()
ENGINES = _available_engines() if HAS_GPU else []
skip_no_gpu = pytest.mark.skipif(not HAS_GPU, reason="no NVIDIA GPU visible through NVML")
skip_no_engine = pytest.mark.skipif(not ENGINES, reason="no inference engine installed")


def _env(tmp_path: Path) -> dict[str, str]:
    env = {
        **os.environ,
        "SERVEPILOT_CACHE_DIR": str(tmp_path / "cache"),
        "SERVEPILOT_STATE_DIR": str(tmp_path / "state"),
        "PYTHONUNBUFFERED": "1",
        "COLUMNS": "160",
    }
    env.pop("SERVEPILOT_FAKE_HARDWARE", None)
    env.pop("SERVEPILOT_ENABLE_FAKE_ENGINE", None)
    env.pop("HF_HUB_OFFLINE", None)
    return env


def _run(
    args: list[str], env: dict[str, str], timeout: float = 600
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "servepilot", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _gpu_memory_used_mib() -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return [int(x.strip()) for x in out.stdout.strip().splitlines() if x.strip()]


@skip_no_gpu
def test_doctor_and_inspect_hardware(tmp_path: Path) -> None:
    env = _env(tmp_path)
    doctor = _run(["doctor", "--json", "--port", str(ephemeral_port())], env)
    payload = json.loads(doctor.stdout)
    labels = {c["label"]: c["status"] for c in payload["checks"]}
    assert labels.get("NVML available") == "ok", labels
    assert any("GPU(s) visible" in label for label in labels)
    hw = json.loads(_run(["inspect", "hardware", "--json"], env).stdout)
    assert hw["gpu_count"] >= 1 and hw["driver_version"] and hw["gpus"][0]["total_memory_bytes"] > 0
    if hw["gpu_count"] >= 2:
        assert hw["topology"]["available"] and hw["topology"]["edges"]


@skip_no_gpu
def test_inspect_model_from_hub(tmp_path: Path) -> None:
    payload = json.loads(_run(["inspect", "model", SMOKE_MODEL, "--json"], _env(tmp_path)).stdout)
    assert payload["weight_size_source"] in ("huggingface_metadata", "safetensors_metadata")
    assert payload["num_hidden_layers"] and payload["revision"]


@skip_no_gpu
@skip_no_engine
def test_plan_real_hardware(tmp_path: Path) -> None:
    result = _run(["plan", SMOKE_MODEL, "--json"], _env(tmp_path))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    candidates = payload["planning"]["candidates"]
    assert candidates and candidates[0]["tensor_parallel_size"] == 1
    gpu_count = payload["hardware"]["gpu_count"]
    if gpu_count >= 2:
        assert any(c["tensor_parallel_size"] == 2 for c in candidates)


@skip_no_gpu
@skip_no_engine
@pytest.mark.parametrize("engine", ENGINES)
def test_serve_smoke(engine: str, tmp_path: Path) -> None:
    """Launch one replica with a real engine, talk to it through ServePilot, stop it, verify cleanup."""
    env = _env(tmp_path)
    port = ephemeral_port()
    baseline = _gpu_memory_used_mib()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "servepilot",
            "-v",
            "serve",
            SMOKE_MODEL,
            "--engine",
            engine,
            "--no-tune",
            "--tp",
            "1",
            "--replicas",
            "1",
            "--gpus",
            "0",
            "--context-length",
            "2048",
            # The chat preset's p95 prompt (2048) + output (768) would not fit this context.
            "--input-tokens",
            "64",
            "--output-tokens",
            "32",
            "--port",
            str(port),
            "--startup-timeout",
            "900",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    output = ""
    try:
        deadline = time.time() + 900
        ready = False
        while time.time() < deadline and proc.poll() is None:
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    ready = True
                    break
            except httpx.HTTPError:
                time.sleep(1)
        if not ready:
            proc.kill()
            output = proc.stdout.read() if proc.stdout else ""
        assert ready, output[-6000:]
        models = httpx.get(f"{base}/v1/models", timeout=10).json()
        assert models["data"][0]["id"] == SMOKE_MODEL
        chat = httpx.post(
            f"{base}/v1/chat/completions",
            json={
                "model": SMOKE_MODEL,
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 8,
                "temperature": 0,
            },
            timeout=120,
        )
        assert chat.status_code == 200, chat.text
        body = chat.json()
        assert body["choices"][0]["message"]["content"] and body["usage"]["completion_tokens"] >= 1
        with httpx.stream(
            "POST",
            f"{base}/v1/chat/completions",
            json={
                "model": SMOKE_MODEL,
                "messages": [{"role": "user", "content": "Count to five."}],
                "max_tokens": 12,
                "stream": True,
            },
            timeout=120,
        ) as stream:
            lines = [line for line in stream.iter_lines() if line.startswith("data:")]
        assert len(lines) >= 3 and lines[-1].strip() == "data: [DONE]"
        completion = httpx.post(
            f"{base}/v1/completions",
            json={"model": SMOKE_MODEL, "prompt": "The capital of France is", "max_tokens": 4},
            timeout=120,
        )
        assert completion.status_code == 200 and completion.json()["choices"][0]["text"]
        status = json.loads(_run(["status", "--json"], env).stdout)
        assert status["running"] is True and status["state"]["engine"] == engine
        stop = json.loads(_run(["stop", "--json"], env, timeout=300).stdout)
        assert stop["stopped"] is True, stop
        proc.wait(timeout=120)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
    time.sleep(5)
    after = _gpu_memory_used_mib()
    assert after[0] <= baseline[0] + 1024, (
        f"GPU memory not released: before {baseline} after {after}"
    )


@skip_no_gpu
@skip_no_engine
def test_tune_real_engine(tmp_path: Path) -> None:
    """Full tuning loop on real hardware with the first available engine (small request counts)."""
    engine = ENGINES[0]
    env = _env(tmp_path)
    cfg = tmp_path / "servepilot.yaml"
    cfg.write_text(
        f"""
model: {SMOKE_MODEL}
engine: {engine}
profile:
  name: custom
  input_tokens_p50: 64
  input_tokens_p95: 128
  output_tokens_p50: 32
  output_tokens_p95: 64
  max_context_tokens: 2048
tuning:
  stage_a_requests: 16
  sweep_requests: 16
  final_multiplier: 2
  memory_tuning: false
  top_k: 1
  startup_timeout_seconds: 900
"""
    )
    result = _run(["tune", "--config", str(cfg), "--json", "--show-pareto"], env, timeout=3600)
    assert result.returncode == 0, result.stderr[-8000:]
    record = json.loads(result.stdout)
    assert record["status"] == "complete"
    winner = record["winner"]
    assert winner["benchmarked"] and winner["final_result"]["successful_requests"] > 0
    assert winner["final_result"]["output_tokens_per_second"] > 0
    assert winner["engine_version"]
    benchmarked = [c for c in record["candidates"] if c["status"] == "benchmarked"]
    assert benchmarked
    time.sleep(5)
    assert all(m < 2048 for m in _gpu_memory_used_mib()), (
        "engine processes must release GPU memory after tuning"
    )
