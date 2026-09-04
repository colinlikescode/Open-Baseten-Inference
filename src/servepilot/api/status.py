"""``/status`` payload with plan, replica and benchmark information (never secrets)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from servepilot.api.app import ServingContext


def status_payload(ctx: ServingContext) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": ctx.model_id,
        "served_model_name": ctx.served_model_name,
        "uptime_seconds": round(ctx.uptime_seconds, 1),
        "router": ctx.router.snapshot(),
    }
    if ctx.selected is not None:
        plan = ctx.selected.plan
        payload.update(
            {
                "objective": ctx.selected.objective.value,
                "engine": plan.engine.value,
                "tensor_parallel_size": plan.tensor_parallel_size,
                "pipeline_parallel_size": plan.pipeline_parallel_size,
                "data_parallel_size": plan.data_parallel_size,
                "replica_count": plan.replica_count,
                "max_concurrency": plan.max_concurrency,
                "max_num_seqs": plan.max_num_seqs,
                "memory_fraction": plan.memory_fraction,
                "context_length": plan.context_length,
                "gpu_groups": plan.gpu_groups,
                "plan_id": plan.id,
                "plan_source": ctx.selected.source,
                "benchmarked": ctx.selected.benchmarked,
                "rationale": ctx.selected.rationale,
            }
        )
        summary = ctx.selected.summary()
        if "benchmark_summary" in summary:
            payload["benchmark_summary"] = summary["benchmark_summary"]
    if ctx.extra_status is not None:
        payload.update(ctx.extra_status())
    return payload
