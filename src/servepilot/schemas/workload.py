"""Workload profile and objective schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Objective(StrEnum):
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    BALANCED = "balanced"


class LatencyConstraints(BaseModel):
    """Hard latency SLOs. A candidate violating any bound is invalid for final selection."""

    max_p95_ttft_ms: float | None = Field(default=None, gt=0)
    max_p95_latency_ms: float | None = Field(default=None, gt=0)
    max_p95_tpot_ms: float | None = Field(default=None, gt=0)

    @property
    def is_empty(self) -> bool:
        return (
            self.max_p95_ttft_ms is None
            and self.max_p95_latency_ms is None
            and self.max_p95_tpot_ms is None
        )


class WorkloadProfile(BaseModel):
    """Describes the traffic ServePilot should optimize for."""

    name: str = "chat"

    input_tokens_p50: int = Field(default=512, ge=1)
    input_tokens_p95: int = Field(default=2048, ge=1)

    output_tokens_p50: int = Field(default=256, ge=1)
    output_tokens_p95: int = Field(default=768, ge=1)

    max_context_tokens: int = Field(default=8192, ge=16)

    expected_concurrency: int | None = Field(default=None, ge=1)
    target_request_rate: float | None = Field(default=None, gt=0)

    shared_prefix_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    streaming: bool = True

    objective: Objective = Objective.THROUGHPUT

    latency_constraints: LatencyConstraints | None = None

    @model_validator(mode="after")
    def _check_ordering(self) -> WorkloadProfile:
        if self.input_tokens_p95 < self.input_tokens_p50:
            raise ValueError("input_tokens_p95 must be >= input_tokens_p50")
        if self.output_tokens_p95 < self.output_tokens_p50:
            raise ValueError("output_tokens_p95 must be >= output_tokens_p50")
        if self.input_tokens_p95 + self.output_tokens_p95 > self.max_context_tokens:
            raise ValueError(
                "input_tokens_p95 + output_tokens_p95 exceeds max_context_tokens "
                f"({self.input_tokens_p95} + {self.output_tokens_p95} > {self.max_context_tokens})"
            )
        return self

    @property
    def p50_sequence_tokens(self) -> int:
        return self.input_tokens_p50 + self.output_tokens_p50

    @property
    def p95_sequence_tokens(self) -> int:
        return self.input_tokens_p95 + self.output_tokens_p95

    @property
    def has_slo(self) -> bool:
        return self.latency_constraints is not None and not self.latency_constraints.is_empty


PRESETS: dict[str, dict[str, int]] = {
    "chat": {
        "input_tokens_p50": 512,
        "input_tokens_p95": 2048,
        "output_tokens_p50": 256,
        "output_tokens_p95": 768,
        "max_context_tokens": 8192,
    },
    "long-context": {
        "input_tokens_p50": 8192,
        "input_tokens_p95": 32768,
        "output_tokens_p50": 256,
        "output_tokens_p95": 1024,
        "max_context_tokens": 65536,
    },
    "decode-heavy": {
        "input_tokens_p50": 256,
        "input_tokens_p95": 1024,
        "output_tokens_p50": 1024,
        "output_tokens_p95": 4096,
        "max_context_tokens": 8192,
    },
}

# Default concurrency used for latency-objective benchmarks when the user gives none.
DEFAULT_LATENCY_CONCURRENCY = 16


def workload_from_preset(
    name: str,
    *,
    objective: Objective = Objective.THROUGHPUT,
    expected_concurrency: int | None = None,
    target_request_rate: float | None = None,
    streaming: bool = True,
    latency_constraints: LatencyConstraints | None = None,
    max_context_tokens: int | None = None,
) -> WorkloadProfile:
    """Build a :class:`WorkloadProfile` from a named preset."""
    if name not in PRESETS:
        raise ValueError(f"unknown workload preset {name!r}; choose from {sorted(PRESETS)}")
    values = dict(PRESETS[name])
    if max_context_tokens is not None:
        values["max_context_tokens"] = max_context_tokens
    return WorkloadProfile(
        name=name,
        objective=objective,
        expected_concurrency=expected_concurrency,
        target_request_rate=target_request_rate,
        streaming=streaming,
        latency_constraints=latency_constraints,
        **values,
    )
