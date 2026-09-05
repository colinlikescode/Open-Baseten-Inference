"""Workload fingerprint and cached-record validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from servepilot.engines.interpreter import parse_version
from servepilot.hardware.fingerprint import hardware_fingerprint
from servepilot.models.fingerprint import model_fingerprint
from servepilot.planner.memory import format_bytes
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.runtime import TuningRecord
from servepilot.schemas.workload import WorkloadProfile

__all__ = [
    "CacheValidation",
    "hardware_fingerprint",
    "model_fingerprint",
    "validate_record",
    "workload_fingerprint",
]


def workload_identity(workload: WorkloadProfile) -> dict[str, Any]:
    slo = workload.latency_constraints
    return {
        "objective": workload.objective.value,
        "input_tokens_p50": workload.input_tokens_p50,
        "input_tokens_p95": workload.input_tokens_p95,
        "output_tokens_p50": workload.output_tokens_p50,
        "output_tokens_p95": workload.output_tokens_p95,
        "max_context_tokens": workload.max_context_tokens,
        "expected_concurrency": workload.expected_concurrency,
        "target_request_rate": workload.target_request_rate,
        "shared_prefix_fraction": workload.shared_prefix_fraction,
        "streaming": workload.streaming,
        "slo": slo.model_dump() if slo is not None else None,
    }


def workload_fingerprint(workload: WorkloadProfile) -> str:
    payload = json.dumps(workload_identity(workload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheValidation:
    valid: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def engine_versions_compatible(recorded: str | None, current: str | None) -> bool:
    """Same major.minor is treated as compatible; anything else is not."""
    if recorded is None or current is None:
        return False
    a, b = parse_version(recorded), parse_version(current)
    return len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2]


def validate_record(
    record: TuningRecord,
    *,
    hardware: HardwareSnapshot,
    model: ModelProfile,
    workload: WorkloadProfile,
    engine_versions: Mapping[str, str],
    gpu_ids: list[int] | None = None,
) -> CacheValidation:
    """Decide whether a cached tuning record can be reused for the current situation."""
    reasons: list[str] = []
    warnings: list[str] = []

    if record.status != "complete" or record.winner is None:
        reasons.append("cached tuning record is incomplete")
    if record.model_fingerprint != model_fingerprint(model):
        reasons.append("model changed (id, revision, config or weights differ)")
    if record.hardware_fingerprint != hardware_fingerprint(hardware, gpu_ids):
        reasons.append("hardware changed (GPU set, memory, topology or driver differ)")
    if record.workload_fingerprint != workload_fingerprint(workload):
        reasons.append(
            "workload profile changed (objective, token distribution, concurrency or SLO differ)"
        )

    if record.winner is not None:
        engine = record.winner.plan.engine.value
        recorded = record.engine_versions.get(engine)
        current = engine_versions.get(engine)
        if current is None:
            reasons.append(f"engine {engine} is no longer available")
        elif not engine_versions_compatible(recorded, current):
            reasons.append(f"engine {engine} version changed ({recorded} → {current})")

        # Enough free memory: every GPU the plan uses must have at least the engine budget free.
        # The plan's fraction is authoritative (memory tuning may have raised it after the
        # static estimate was made).
        plan = record.winner.plan
        est = plan.estimated_memory
        for gpu_id in plan.gpu_ids:
            try:
                gpu = hardware.gpu(gpu_id)
            except KeyError:
                reasons.append(f"GPU {gpu_id} used by the cached plan is not present")
                continue
            if plan.memory_fraction is not None:
                required = int(plan.memory_fraction * gpu.total_memory_bytes)
            elif est is not None:
                required = est.engine_budget_bytes
            else:
                required = int(0.9 * gpu.total_memory_bytes)
            if gpu.free_memory_bytes < required:
                reasons.append(
                    f"GPU {gpu_id} has {format_bytes(gpu.free_memory_bytes)} free but the cached plan needs "
                    f"{format_bytes(required)}"
                )
    return CacheValidation(valid=not reasons, reasons=reasons, warnings=warnings)
