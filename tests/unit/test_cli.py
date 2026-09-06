"""CLI commands that need no engines: version, doctor, inspect, plan, cache, status/stop, serve --dry-run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from servepilot import __version__
from servepilot.cli.app import app
from servepilot.constants import GIB
from tests.conftest import make_local_model

runner = CliRunner()


@pytest.fixture
def fake_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("SERVEPILOT_FAKE_HARDWARE", "h100x4")
    monkeypatch.setenv("SERVEPILOT_ENABLE_FAKE_ENGINE", "1")
    monkeypatch.setenv("SERVEPILOT_VLLM_PYTHON", "/nonexistent/python")
    monkeypatch.setenv("SERVEPILOT_SGLANG_PYTHON", "/nonexistent/python")
    return make_local_model(tmp_path, "llama3_8b", 16 * GIB)


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0 and __version__ in result.stdout
    result = runner.invoke(app, ["version", "--json"])
    assert json.loads(result.stdout)["version"] == __version__


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    for command in (
        "doctor",
        "inspect",
        "plan",
        "tune",
        "serve",
        "benchmark",
        "status",
        "stop",
        "cache",
        "version",
    ):
        assert command in result.stdout


def test_doctor_json_with_fake_hardware(fake_env: Path) -> None:
    result = runner.invoke(app, ["doctor", "--json", "--port", "65000"])
    payload = json.loads(result.stdout)
    labels = [c["label"] for c in payload["checks"]]
    assert any("4 GPU(s) visible" in label for label in labels)
    assert any("fake engine" in label for label in labels)
    assert payload["status"] in ("ok", "warn")
    human = runner.invoke(app, ["doctor", "--port", "65000"])
    assert human.exit_code == 0 and "ServePilot doctor" in human.stdout


def test_doctor_fails_without_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVEPILOT_VLLM_PYTHON", "/nonexistent/python")
    monkeypatch.setenv("SERVEPILOT_SGLANG_PYTHON", "/nonexistent/python")
    import sys
    import types

    mod = types.ModuleType("pynvml")

    class Err(Exception):
        pass

    def init() -> None:
        raise Err("no driver")

    mod.NVMLError = Err  # type: ignore[attr-defined]
    mod.nvmlInit = init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", mod)
    result = runner.invoke(app, ["doctor", "--json"])
    payload = json.loads(result.stdout)
    assert payload["status"] == "fail" and result.exit_code == 3
    assert any("NVML" in c["label"] for c in payload["checks"] if c["status"] == "fail")


def test_inspect_hardware(fake_env: Path) -> None:
    result = runner.invoke(app, ["inspect", "hardware", "--json"])
    payload = json.loads(result.stdout)
    assert payload["gpu_count"] == 4 and payload["topology"]["edges"]
    human = runner.invoke(app, ["inspect", "hardware"])
    assert human.exit_code == 0 and "NVIDIA H100" in human.stdout


def test_inspect_model_local(fake_env: Path) -> None:
    result = runner.invoke(app, ["inspect", "model", str(fake_env), "--json"])
    payload = json.loads(result.stdout)
    assert payload["weight_bytes"] == 16 * GIB and payload["weight_size_source"] == "local_files"
    human = runner.invoke(app, ["inspect", "model", str(fake_env)])
    assert "16.0 GiB (local_files)" in human.stdout


def test_inspect_model_missing_path_is_clean_error(fake_env: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["inspect", "model", str(tmp_path / "not-a-model")])
    assert result.exit_code == 4
    assert "error" in result.output.lower()


def test_plan_json_and_human(fake_env: Path) -> None:
    result = runner.invoke(app, ["plan", str(fake_env), "--engine", "fake", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    plans = payload["planning"]["candidates"]
    assert [p["tensor_parallel_size"] for p in plans] == [1, 2, 4]
    assert payload["planning"]["estimated_minimum_tp"] == 1
    human = runner.invoke(app, ["plan", str(fake_env), "--engine", "fake", "--tp", "2"])
    assert human.exit_code == 0 and "TP2 × 2 replicas" in human.stdout and "Why" in human.stdout


def test_plan_configuration_errors(fake_env: Path) -> None:
    bad_tp = runner.invoke(app, ["plan", str(fake_env), "--engine", "fake", "--tp", "3"])
    assert bad_tp.exit_code == 2 and "--tp 3" in bad_tp.output
    no_model = runner.invoke(app, ["plan"])
    assert no_model.exit_code == 2
    bad_engine = runner.invoke(app, ["plan", str(fake_env), "--engine", "vllm"])
    assert bad_engine.exit_code == 5


def test_plan_with_config_file(fake_env: Path, tmp_path: Path) -> None:
    cfg = tmp_path / "servepilot.yaml"
    cfg.write_text(
        f"model: {fake_env}\nengine: fake\nobjective: latency\nprofile:\n  name: decode-heavy\n  expected_concurrency: 8\nconstraints:\n  replicas: 2\n  tp: 2\n"
    )
    result = runner.invoke(app, ["plan", "--config", str(cfg), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert (
        payload["workload"]["objective"] == "latency"
        and payload["workload"]["name"] == "decode-heavy"
    )
    assert [p["replica_count"] for p in payload["planning"]["candidates"]] == [2]


def test_serve_dry_run_with_no_tune(fake_env: Path) -> None:
    result = runner.invoke(
        app,
        ["serve", str(fake_env), "--engine", "fake", "--no-tune", "--dry-run", "--port", "65010"],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.stdout and "UNBENCHMARKED" in result.stdout
    assert "fake_openai" in result.stdout


def test_serve_dry_run_never_launches_tuning(
    fake_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def forbidden_launch(*args, **kwargs):
        raise AssertionError("dry-run must not launch any engine process")

    monkeypatch.setattr("servepilot.engines.process.LocalLauncher.launch", forbidden_launch)
    result = runner.invoke(app, ["serve", str(fake_env), "--engine", "fake", "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    selected = json.loads(result.stdout)["selected"]
    assert selected["source"] == "heuristic" and selected["benchmarked"] is False
    as_json = runner.invoke(
        app, ["serve", str(fake_env), "--engine", "fake", "--no-tune", "--dry-run", "--json"]
    )
    payload = json.loads(as_json.stdout)
    assert payload["selected"]["source"] == "heuristic" and len(payload["launches"]) == 4


def test_cache_commands_empty(fake_env: Path) -> None:
    listed = runner.invoke(app, ["cache", "list"])
    assert listed.exit_code == 0 and "No tuning records" in listed.stdout
    assert json.loads(runner.invoke(app, ["cache", "list", "--json"]).stdout) == []
    cleared = runner.invoke(app, ["cache", "clear", "--json"])
    assert json.loads(cleared.stdout) == {"removed": 0}
    missing = runner.invoke(app, ["cache", "show", "nope"])
    assert missing.exit_code == 9


def test_status_and_stop_without_deployment(fake_env: Path) -> None:
    status = runner.invoke(app, ["status"])
    assert status.exit_code == 0 and "No ServePilot deployment" in status.stdout
    assert json.loads(runner.invoke(app, ["status", "--json"]).stdout) == {"running": False}
    stop = runner.invoke(app, ["stop", "--json"])
    assert json.loads(stop.stdout)["stopped"] is False


def test_heterogeneous_hardware_error(monkeypatch: pytest.MonkeyPatch, fake_env: Path) -> None:
    monkeypatch.setenv("SERVEPILOT_FAKE_HARDWARE", "mixed_h100_a100")
    result = runner.invoke(app, ["plan", str(fake_env), "--engine", "fake"])
    assert result.exit_code == 3 and "--gpus 0,1" in result.output
    ok = runner.invoke(app, ["plan", str(fake_env), "--engine", "fake", "--gpus", "0,1", "--json"])
    assert ok.exit_code == 0 and json.loads(ok.stdout)["planning"]["selected_gpu_ids"] == [0, 1]


def test_unexpected_error_is_summarised(monkeypatch: pytest.MonkeyPatch, fake_env: Path) -> None:
    monkeypatch.setenv("SERVEPILOT_FAKE_HARDWARE", "does-not-exist")
    result = runner.invoke(app, ["inspect", "hardware"])
    assert result.exit_code == 1 and "unexpected error" in result.output and "-vv" in result.output
