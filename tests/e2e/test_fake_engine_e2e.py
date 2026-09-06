"""End-to-end: real subprocess engines (fake OpenAI backends), real tuner, real deployment."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil
import pytest

from servepilot.benchmark.evaluator import LaunchingEvaluator
from servepilot.benchmark.tuner import Tuner, TuningSettings
from servepilot.constants import GIB
from servepilot.engines.process import LocalLauncher
from servepilot.engines.registry import EngineRegistry
from servepilot.models.inspector import inspect_local_model
from servepilot.models.tokenizer import ApproximateTokenizer
from servepilot.planner.candidates import generate_candidates
from servepilot.runtime.ports import PortAllocator, ephemeral_port
from servepilot.schemas.benchmark import FailureType
from servepilot.schemas.workload import WorkloadProfile
from servepilot.testing import fake_hardware as fh
from servepilot.testing.fake_engine import FakeEngine, FakeEngineBehavior, FakePlanBehavior
from servepilot.testing.fake_hardware import FakeHardwareProvider
from tests.conftest import make_local_model

pytestmark = pytest.mark.e2e

SMALL_WORKLOAD = WorkloadProfile(
    name="tiny",
    input_tokens_p50=16,
    input_tokens_p95=32,
    output_tokens_p50=8,
    output_tokens_p95=16,
    max_context_tokens=1024,
)
FAST_SETTINGS = TuningSettings(
    stage_a_requests=8,
    sweep_requests=8,
    final_multiplier=1,
    sweep_start=4,
    sweep_max=32,
    memory_tuning=False,
    top_k=1,
)


def _child_pids_of_fake_backends() -> set[int]:
    pids: set[int] = set()
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmd = proc.info["cmdline"] or []
        except psutil.Error:
            continue
        if any("servepilot.testing.fake_openai" in part for part in cmd):
            pids.add(proc.pid)
    return pids


async def _tune(
    model_dir: Path,
    behavior: FakeEngineBehavior,
    workload: WorkloadProfile = SMALL_WORKLOAD,
    gpus: int = 2,
    settings: TuningSettings = FAST_SETTINGS,
):
    snapshot = fh.h100(gpus)
    provider = FakeHardwareProvider(snapshot)
    model = inspect_local_model(model_dir)
    engine = FakeEngine(behavior)
    registry = EngineRegistry([engine])
    planning = generate_candidates(snapshot, model, workload, [engine])
    launcher = LocalLauncher()
    evaluator = LaunchingEvaluator(
        registry=registry,
        launcher=launcher,
        ports=PortAllocator(33000, 33999),
        hardware_provider=provider,
        hardware=snapshot,
        model=model,
        workload=workload,
        tokenizer=ApproximateTokenizer(),
        startup_timeout=60,
        stagger_seconds=0,
    )
    try:
        outcome = await Tuner(evaluator, workload, settings).tune(planning)
    finally:
        await launcher.shutdown_all()
    return outcome, planning


async def test_full_tuning_loop_with_real_subprocesses(tmp_path: Path) -> None:
    model_dir = make_local_model(tmp_path, "llama3_8b", 16 * GIB)
    before = _child_pids_of_fake_backends()
    behavior = FakeEngineBehavior(base_tpot_ms=1.0, base_ttft_ms=2.0, base_capacity=8)
    outcome, _planning = await _tune(model_dir, behavior)
    winner = outcome.winner
    assert (
        winner.benchmarked
        and winner.final_result is not None
        and winner.final_result.successful_requests > 0
    )
    assert winner.plan.max_concurrency is not None and winner.plan.max_concurrency >= 4
    assert {e.stage for e in outcome.evaluations} >= {"structural", "concurrency", "final"}
    assert all(e.status == "benchmarked" for e in outcome.evaluations)
    assert (
        winner.final_result.ttft_p95_ms is not None and winner.final_result.tpot_p95_ms is not None
    )
    # Every temporary engine process must be gone.
    await asyncio.sleep(0.5)
    assert _child_pids_of_fake_backends() <= before


async def test_oom_candidate_is_skipped_and_second_wins(tmp_path: Path) -> None:
    model_dir = make_local_model(tmp_path, "llama3_8b", 16 * GIB)
    behavior = FakeEngineBehavior(
        base_tpot_ms=1.0,
        base_ttft_ms=2.0,
        base_capacity=8,
        plans={"tp1": FakePlanBehavior(startup="oom", startup_delay_s=0.1)},
    )
    outcome, _ = await _tune(model_dir, behavior)
    assert outcome.winner.plan.tensor_parallel_size == 2
    failed = [e for e in outcome.evaluations if e.status == "failed"]
    assert failed and failed[0].failure is not None
    assert (
        failed[0].failure.type == FailureType.OOM
        and "out of memory" in failed[0].failure.stderr_tail.lower()
    )
    assert any("failed to launch (oom)" in line for line in outcome.winner.rationale)


async def test_startup_timeout_and_crash_are_classified(tmp_path: Path) -> None:
    model_dir = make_local_model(tmp_path, "llama3_8b", 16 * GIB)
    behavior = FakeEngineBehavior(
        base_tpot_ms=1.0,
        base_ttft_ms=2.0,
        base_capacity=8,
        plans={
            "tp1": FakePlanBehavior(startup="hang", startup_delay_s=0.0),
            "tp2": FakePlanBehavior(startup="crash", startup_delay_s=0.1),
        },
    )
    snapshot = fh.h100(2)
    model = inspect_local_model(model_dir)
    engine = FakeEngine(behavior)
    planning = generate_candidates(snapshot, model, SMALL_WORKLOAD, [engine])
    launcher = LocalLauncher()
    evaluator = LaunchingEvaluator(
        registry=EngineRegistry([engine]),
        launcher=launcher,
        ports=PortAllocator(33000, 33999),
        hardware_provider=FakeHardwareProvider(snapshot),
        hardware=snapshot,
        model=model,
        workload=SMALL_WORKLOAD,
        tokenizer=ApproximateTokenizer(),
        startup_timeout=2.0,
        stagger_seconds=0,
    )
    tuner = Tuner(evaluator, SMALL_WORKLOAD, FAST_SETTINGS)
    from servepilot.exceptions import NoViablePlanError

    try:
        with pytest.raises(NoViablePlanError):
            await tuner.tune(planning)
    finally:
        await launcher.shutdown_all()
    types = {e.plan.tensor_parallel_size: e.failure.type for e in tuner.evaluations if e.failure}
    assert types[1] == FailureType.STARTUP_TIMEOUT and types[2] == FailureType.ENGINE_CRASH
    await asyncio.sleep(0.3)
    assert launcher.tracked() == []


def _env(tmp_path: Path, hardware: str = "h100x2", behavior: Path | None = None) -> dict[str, str]:
    env = {
        **os.environ,
        "SERVEPILOT_FAKE_HARDWARE": hardware,
        "SERVEPILOT_ENABLE_FAKE_ENGINE": "1",
        "SERVEPILOT_CACHE_DIR": str(tmp_path / "cache"),
        "SERVEPILOT_STATE_DIR": str(tmp_path / "state"),
        "SERVEPILOT_BACKEND_PORT_START": "34000",
        "SERVEPILOT_BACKEND_PORT_END": "34999",
        "SERVEPILOT_VLLM_PYTHON": "/nonexistent",
        "SERVEPILOT_SGLANG_PYTHON": "/nonexistent",
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "COLUMNS": "160",
    }
    if behavior is not None:
        env["SERVEPILOT_FAKE_ENGINE_BEHAVIOR"] = str(behavior)
    return env


def _config(tmp_path: Path, model_dir: Path) -> Path:
    cfg = tmp_path / "servepilot.yaml"
    cfg.write_text(
        f"""
model: {model_dir}
engine: fake
profile:
  name: custom
  input_tokens_p50: 16
  input_tokens_p95: 32
  output_tokens_p50: 8
  output_tokens_p95: 16
  max_context_tokens: 1024
tuning:
  stage_a_requests: 8
  sweep_requests: 8
  final_multiplier: 1
  memory_tuning: false
  top_k: 1
"""
    )
    return cfg


def test_cli_tune_then_serve_uses_cache(tmp_path: Path) -> None:
    model_dir = make_local_model(tmp_path, "llama3_8b", 16 * GIB)
    behavior = tmp_path / "behavior.json"
    behavior.write_text(json.dumps({"base_tpot_ms": 1.0, "base_ttft_ms": 2.0, "base_capacity": 8}))
    env = _env(tmp_path, behavior=behavior)
    cfg = _config(tmp_path, model_dir)
    # A forced engine budget must remain fixed even when memory tuning is enabled.
    cfg.write_text(
        cfg.read_text().replace("memory_tuning: false", "memory_tuning: true")
        + "constraints:\n  memory_fraction: 0.8\n"
    )

    tune = subprocess.run(
        [
            sys.executable,
            "-m",
            "servepilot",
            "tune",
            "--config",
            str(cfg),
            "--json",
            "--show-pareto",
            "-o",
            str(tmp_path / "out.json"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert tune.returncode == 0, tune.stderr[-4000:]
    record = json.loads(tune.stdout)
    assert record["status"] == "complete" and record["winner"]["benchmarked"] is True
    assert record["winner"]["plan"]["memory_fraction"] == 0.8
    assert all(candidate["stage"] != "memory" for candidate in record["candidates"])
    assert (tmp_path / "out.json").exists()
    assert "Pareto" in tune.stderr
    cache_files = list((tmp_path / "cache" / "tuning").glob("*.json"))
    assert len(cache_files) == 1

    listed = subprocess.run(
        [sys.executable, "-m", "servepilot", "cache", "list", "--json"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    entries = json.loads(listed.stdout)
    assert len(entries) == 1 and entries[0]["status"] == "complete"
    shown = subprocess.run(
        [sys.executable, "-m", "servepilot", "cache", "show", entries[0]["key"]],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert shown.returncode == 0 and "Selected plan" in shown.stdout

    # serve --dry-run must reuse the cached winner (no tuning).
    dry = subprocess.run(
        [sys.executable, "-m", "servepilot", "serve", "--config", str(cfg), "--dry-run", "--json"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert dry.returncode == 0, dry.stderr[-4000:]
    payload = json.loads(dry.stdout)
    assert payload["selected"]["source"] == "cached"
    assert "Using cached tuning result" in dry.stderr

    capped = subprocess.run(
        [
            sys.executable,
            "-m",
            "servepilot",
            "serve",
            "--config",
            str(cfg),
            "--max-concurrency",
            "4",
            "--dry-run",
            "--json",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert capped.returncode == 0, capped.stderr[-4000:]
    selection = json.loads(capped.stdout)["selected"]
    assert selection["source"] == "heuristic" and selection["plan"]["max_num_seqs"] == 4

    # A different workload must not reuse the cached plan.
    other = subprocess.run(
        [
            sys.executable,
            "-m",
            "servepilot",
            "serve",
            "--config",
            str(cfg),
            "--profile",
            "decode-heavy",
            "--context-length",
            "1024",
            "--input-tokens",
            "16",
            "--output-tokens",
            "8",
            "--dry-run",
            "--json",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert other.returncode == 0, other.stderr[-4000:]
    assert json.loads(other.stdout)["selected"]["source"] == "heuristic"


def test_cli_serve_end_to_end(tmp_path: Path) -> None:
    model_dir = make_local_model(tmp_path, "llama3_8b", 16 * GIB)
    env = _env(tmp_path)
    cfg = _config(tmp_path, model_dir)
    port = ephemeral_port()
    before = _child_pids_of_fake_backends()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "servepilot",
            "serve",
            "--config",
            str(cfg),
            "--no-tune",
            "--port",
            str(port),
            "--served-model-name",
            "demo",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 120
        healthy = False
        while time.time() < deadline and proc.poll() is None:
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    healthy = True
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        assert healthy, proc.stdout.read() if proc.stdout else ""
        health = httpx.get(f"{base}/health", timeout=5).json()
        assert (
            health["healthy_replicas"] == 2
            and health["total_replicas"] == 2
            and health["model"] == "demo"
        )
        chat = httpx.post(
            f"{base}/v1/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            },
            timeout=30,
        )
        assert chat.status_code == 200 and chat.json()["usage"]["completion_tokens"] == 4
        with httpx.stream(
            "POST",
            f"{base}/v1/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
                "stream": True,
            },
            timeout=30,
        ) as stream:
            lines = [line for line in stream.iter_lines() if line.startswith("data:")]
        assert lines[-1].strip() == "data: [DONE]" and len(lines) >= 4
        models = httpx.get(f"{base}/v1/models", timeout=5).json()
        assert models["data"][0]["id"] == "demo"
        status_payload = httpx.get(f"{base}/status", timeout=5).json()
        assert status_payload["replica_count"] == 2 and status_payload["plan_source"] == "heuristic"
        assert "servepilot_requests_total" in httpx.get(f"{base}/metrics", timeout=5).text

        cli_status = subprocess.run(
            [sys.executable, "-m", "servepilot", "status", "--json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        payload = json.loads(cli_status.stdout)
        assert (
            payload["running"] is True
            and payload["state"]["replica_count"] == 2
            and payload["status"]["router"]["healthy_replicas"] == 2
        )
        human = subprocess.run(
            [sys.executable, "-m", "servepilot", "status"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert "ServePilot status" in human.stdout

        # A second deployment must be refused while this one runs.
        second = subprocess.run(
            [
                sys.executable,
                "-m",
                "servepilot",
                "serve",
                "--config",
                str(cfg),
                "--no-tune",
                "--port",
                str(ephemeral_port()),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert second.returncode == 7 and "already running" in second.stdout + second.stderr

        stop = subprocess.run(
            [sys.executable, "-m", "servepilot", "stop", "--json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert json.loads(stop.stdout)["stopped"] is True, stop.stdout
        proc.wait(timeout=60)
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    time.sleep(0.5)
    assert _child_pids_of_fake_backends() <= before, "engine processes leaked after stop"
    assert not (tmp_path / "state" / "runtime.json").exists()


def test_ctrl_c_cleans_up_engines(tmp_path: Path) -> None:
    model_dir = make_local_model(tmp_path, "llama3_8b", 16 * GIB)
    env = _env(tmp_path, hardware="h100x1")
    cfg = _config(tmp_path, model_dir)
    port = ephemeral_port()
    before = _child_pids_of_fake_backends()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "servepilot",
            "serve",
            "--config",
            str(cfg),
            "--no-tune",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        assert _child_pids_of_fake_backends() - before, "engine should be running"
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    time.sleep(0.5)
    assert _child_pids_of_fake_backends() <= before
    assert not (tmp_path / "state" / "runtime.json").exists()


def test_cli_benchmark_against_running_backend(tmp_path: Path) -> None:
    from tests.integration.conftest import start_backend

    backend = start_backend(ttft_ms=1, tpot_ms=1)
    try:
        env = _env(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "servepilot",
                "benchmark",
                backend.base_url,
                "--concurrency",
                "4",
                "--num-requests",
                "8",
                "--input-tokens",
                "16",
                "--output-tokens",
                "8",
                "--json",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr[-3000:]
        payload = json.loads(result.stdout)
        assert payload["successful_requests"] == 8 and payload["output_tokens_per_second"] > 0
        human = subprocess.run(
            [
                sys.executable,
                "-m",
                "servepilot",
                "benchmark",
                backend.base_url,
                "--concurrency",
                "2",
                "--num-requests",
                "4",
                "--input-tokens",
                "16",
                "--output-tokens",
                "8",
                "--completions",
                "--no-stream",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert human.returncode == 0 and "Benchmark result" in human.stdout
    finally:
        backend.stop()
