"""Fake inference engine adapter.

It launches :mod:`servepilot.testing.fake_openai` as a real subprocess, so the process supervisor,
port allocation, readiness probing, benchmarking, routing and cleanup paths are exercised exactly
as with vLLM/SGLang. Behaviour (startup failures, latency, capacity) is configured per plan via
:class:`FakeEngineBehavior` so tests can script scenarios such as "TP=1 OOMs, TP=2 succeeds".
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from servepilot.engines.base import InferenceEngine, LaunchSpec, SupportResult, redacted_command
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidatePlan, EngineName

StartupMode = Literal["ok", "oom", "crash", "hang", "unsupported", "nccl"]


class FakePlanBehavior(BaseModel):
    startup: StartupMode = "ok"
    startup_delay_s: float = 0.2
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    capacity: int | None = None
    error_rate: float | None = None


class FakeEngineBehavior(BaseModel):
    """Behaviour lookup: ``plans`` keys match ``tp{N}``, ``tp{N}-x{R}`` or the full plan id."""

    default: FakePlanBehavior = Field(default_factory=FakePlanBehavior)
    plans: dict[str, FakePlanBehavior] = Field(default_factory=dict)
    # Baseline per-replica performance model used when a plan does not override it.
    base_ttft_ms: float = 30.0
    base_tpot_ms: float = 6.0
    base_capacity: int = 24
    # Scaling with tensor parallelism: tpot ∝ tp ** -tp_speedup_exponent, capacity ∝ tp.
    tp_speedup_exponent: float = 0.5
    unsupported_model_types: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None) -> FakeEngineBehavior:
        if path is None:
            return cls()
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def for_plan(self, plan: CandidatePlan) -> FakePlanBehavior:
        keys = [
            plan.id,
            f"tp{plan.tensor_parallel_size}-x{plan.replica_count}",
            f"tp{plan.tensor_parallel_size}",
        ]
        for key in keys:
            if key in self.plans:
                merged = self.default.model_dump()
                merged.update(
                    {k: v for k, v in self.plans[key].model_dump().items() if v is not None}
                )
                return FakePlanBehavior.model_validate(merged)
        return self.default

    def performance(self, plan: CandidatePlan) -> tuple[float, float, int]:
        b = self.for_plan(plan)
        tp = plan.tensor_parallel_size
        ttft = b.ttft_ms if b.ttft_ms is not None else self.base_ttft_ms
        tpot = (
            b.tpot_ms
            if b.tpot_ms is not None
            else self.base_tpot_ms / (tp**self.tp_speedup_exponent)
        )
        capacity = b.capacity if b.capacity is not None else self.base_capacity * tp
        return ttft, tpot, capacity


class FakeEngine(InferenceEngine):
    engine_name = EngineName.FAKE

    def __init__(
        self, behavior: FakeEngineBehavior | None = None, *, version: str = "0.0.1-fake"
    ) -> None:
        self.behavior = behavior or FakeEngineBehavior()
        self._version = version
        self.launch_specs: list[LaunchSpec] = []

    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return True

    def version(self) -> str | None:
        return self._version

    def supports_expert_parallel(self, model: ModelProfile) -> bool:
        return model.is_moe

    def supports_pipeline_parallel(self) -> bool:
        return True

    def supports_ray_backend(self) -> bool:
        return True

    def default_max_num_seqs(self) -> int:
        return 128

    def supports(self, model: ModelProfile, plan: CandidatePlan) -> SupportResult:
        if model.model_type and model.model_type in self.behavior.unsupported_model_types:
            return SupportResult(
                supported=False,
                confidence="high",
                reasons=[f"fake engine is configured to reject model type {model.model_type!r}"],
            )
        if plan.dp_attention_enabled:
            return SupportResult(
                supported=False, confidence="high", reasons=["fake engine has no DP attention"]
            )
        return SupportResult(supported=True, confidence="high")

    def build_launch_spec(
        self,
        model: ModelProfile,
        plan: CandidatePlan,
        *,
        replica_index: int,
        host: str,
        port: int,
        served_model_name: str | None = None,
        trust_remote_code: bool = False,
        node: tuple[str, str] | None = None,
        local_gpu_ids: list[int] | None = None,
    ) -> LaunchSpec:
        behavior = self.behavior.for_plan(plan)
        ttft, tpot, capacity = self.behavior.performance(plan)
        gpu_ids = plan.gpu_groups[replica_index]
        device_ids = local_gpu_ids if local_gpu_ids is not None else gpu_ids
        args = [
            "-m",
            "servepilot.testing.fake_openai",
            "--host",
            host,
            "--port",
            str(port),
            "--model",
            served_model_name or model.model_id,
            "--ttft-ms",
            f"{ttft:g}",
            "--tpot-ms",
            f"{tpot:g}",
            "--capacity",
            str(capacity),
            "--max-model-len",
            str(plan.context_length),
            "--startup-mode",
            behavior.startup,
            "--startup-delay",
            f"{behavior.startup_delay_s:g}",
            "--seed",
            str(replica_index),
        ]
        if behavior.error_rate:
            args += ["--error-rate", f"{behavior.error_rate:g}"]
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", "/"),
            "PYTHONUNBUFFERED": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in device_ids),
        }
        if "PYTHONPATH" in os.environ:
            env["PYTHONPATH"] = os.environ["PYTHONPATH"]
        spec = LaunchSpec(
            executable=sys.executable,
            args=args,
            env=env,
            host=host,
            port=port,
            gpu_ids=list(gpu_ids),
            redacted_display_command=redacted_command([sys.executable, *args], env),
            replica_id=f"{plan.id}-r{replica_index}",
            node_id=node[0] if node else None,
            node_ip=node[1] if node else None,
        )
        self.launch_specs.append(spec)
        return spec
