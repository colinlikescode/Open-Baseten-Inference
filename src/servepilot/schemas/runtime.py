"""Runtime state and persisted tuning record schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from servepilot.constants import RUNTIME_STATE_SCHEMA_VERSION, TUNING_RECORD_SCHEMA_VERSION
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidateEvaluation, SelectedPlan
from servepilot.schemas.workload import WorkloadProfile


class ReplicaStatus(StrEnum):
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


class ReplicaState(BaseModel):
    """Routing-relevant state for one backend replica."""

    id: str
    base_url: str
    gpu_ids: list[int] = Field(default_factory=list)
    status: ReplicaStatus = ReplicaStatus.STARTING
    inflight_requests: int = 0
    total_requests: int = 0
    failures: int = 0
    consecutive_health_failures: int = 0
    restarts: int = 0
    pid: int | None = None
    last_error: str | None = None

    @property
    def healthy(self) -> bool:
        return self.status == ReplicaStatus.HEALTHY


class ChildProcessRecord(BaseModel):
    pid: int
    create_time: float
    replica_id: str
    port: int
    gpu_ids: list[int] = Field(default_factory=list)


class RuntimeState(BaseModel):
    """Persisted so ``servepilot status`` / ``servepilot stop`` can find a running deployment."""

    schema_version: int = RUNTIME_STATE_SCHEMA_VERSION
    servepilot_pid: int
    servepilot_create_time: float
    servepilot_version: str
    model: str
    served_model_name: str
    public_host: str
    public_port: int
    backend_ports: list[int] = Field(default_factory=list)
    children: list[ChildProcessRecord] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    plan_id: str
    engine: str
    tensor_parallel_size: int
    replica_count: int
    gpu_groups: list[list[int]] = Field(default_factory=list)


class TuningRecord(BaseModel):
    """Complete, reproducible record of one tuning session."""

    schema_version: int = TUNING_RECORD_SCHEMA_VERSION
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    servepilot_version: str

    hardware_fingerprint: str
    model_fingerprint: str
    workload_fingerprint: str
    engine_versions: dict[str, str] = Field(default_factory=dict)

    hardware_snapshot: HardwareSnapshot
    model_profile: ModelProfile
    workload_profile: WorkloadProfile

    seed: int
    status: Literal["in_progress", "complete", "failed"] = "in_progress"

    candidates: list[CandidateEvaluation] = Field(default_factory=list)
    winner: SelectedPlan | None = None
    notes: list[str] = Field(default_factory=list)
    tuning_settings: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return make_record_key(
            self.hardware_fingerprint, self.model_fingerprint, self.workload_fingerprint
        )


def make_record_key(hardware_fp: str, model_fp: str, workload_fp: str) -> str:
    """Stable cache key from the three fingerprints."""
    return f"{hardware_fp[:12]}-{model_fp[:12]}-{workload_fp[:12]}"
