"""Shared fixtures: fake hardware, model profiles, temporary cache/state directories."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from servepilot.constants import GIB
from servepilot.models.inspector import inspect_local_model, normalize_config
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.workload import Objective, WorkloadProfile, workload_from_preset
from servepilot.settings import ServePilotSettings
from servepilot.testing import fake_hardware as fh

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_config(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / "model_configs" / f"{name}.json").read_text())


def make_local_model(
    root: Path,
    config_name: str,
    weight_bytes: int,
    *,
    shards: int = 2,
    extra_files: dict[str, int] | None = None,
) -> Path:
    """Create a fake model directory with sparse safetensors files of the requested total size."""
    model_dir = root / config_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(load_config(config_name)))
    (model_dir / "tokenizer_config.json").write_text("{}")
    per = weight_bytes // shards
    for i in range(shards):
        path = model_dir / f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        with path.open("wb") as fh_:
            fh_.truncate(per)
    for name, size in (extra_files or {}).items():
        with (model_dir / name).open("wb") as fh_:
            fh_.truncate(size)
    return model_dir


def profile_from_config(
    name: str, *, weight_bytes: int | None, source: str = "local_files"
) -> ModelProfile:
    profile = normalize_config(name, load_config(name), tokenizer_available=True)
    if weight_bytes is not None:
        profile.weight_bytes = weight_bytes
        profile.weight_size_source = source  # type: ignore[assignment]
    return profile


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVEPILOT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SERVEPILOT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("SERVEPILOT_FAKE_HARDWARE", raising=False)
    monkeypatch.delenv("SERVEPILOT_ENABLE_FAKE_ENGINE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture
def settings(tmp_path: Path) -> ServePilotSettings:
    return ServePilotSettings(
        cache_dir=tmp_path / "cache",
        state_dir=tmp_path / "state",
        backend_port_start=31000,
        backend_port_end=31999,
    )


@pytest.fixture
def h100x1() -> HardwareSnapshot:
    return fh.h100x1()


@pytest.fixture
def h100x2() -> HardwareSnapshot:
    return fh.h100x2()


@pytest.fixture
def h100x4() -> HardwareSnapshot:
    return fh.h100x4()


@pytest.fixture
def h100x8() -> HardwareSnapshot:
    return fh.h100x8()


@pytest.fixture
def b200x8() -> HardwareSnapshot:
    return fh.b200x8()


@pytest.fixture
def mixed_hardware() -> HardwareSnapshot:
    return fh.mixed_h100_a100()


@pytest.fixture
def dense_8b() -> ModelProfile:
    """Llama-3-8B-like dense GQA model, 16 GB of bf16 weights."""
    return profile_from_config("llama3_8b", weight_bytes=16 * GIB)


@pytest.fixture
def dense_32b() -> ModelProfile:
    """Qwen3-32B-like dense model, ~61 GiB bf16 weights (fits one 80 GB GPU only at short context)."""
    return profile_from_config("qwen3_32b", weight_bytes=int(61 * GIB))


@pytest.fixture
def dense_70b() -> ModelProfile:
    """Llama-3-70B-like dense model, ~131 GiB bf16 weights (needs TP>=2 on 80 GB GPUs)."""
    return profile_from_config("llama3_70b", weight_bytes=int(131 * GIB))


@pytest.fixture
def moe_8x7b() -> ModelProfile:
    return profile_from_config("mixtral_8x7b", weight_bytes=int(87 * GIB))


@pytest.fixture
def deepseek_v3() -> ModelProfile:
    return profile_from_config("deepseek_v3", weight_bytes=int(642 * GIB))


@pytest.fixture
def quantized_awq() -> ModelProfile:
    return profile_from_config("llama3_70b_awq", weight_bytes=int(37 * GIB))


@pytest.fixture
def chat_workload() -> WorkloadProfile:
    return workload_from_preset("chat")


@pytest.fixture
def latency_workload() -> WorkloadProfile:
    return workload_from_preset("chat", objective=Objective.LATENCY, expected_concurrency=16)


@pytest.fixture
def fake_model_dir(tmp_path: Path) -> Path:
    return make_local_model(tmp_path, "llama3_8b", 16 * GIB)


@pytest.fixture
def fake_model_profile(fake_model_dir: Path) -> ModelProfile:
    return inspect_local_model(fake_model_dir)


@pytest.fixture
def chdir_tmp(tmp_path: Path) -> Iterator[Path]:
    old = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(old)
