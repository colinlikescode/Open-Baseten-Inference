"""``/health`` payload: healthy only when at least one replica can take traffic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from servepilot.api.app import ServingContext


def health_payload(ctx: ServingContext) -> tuple[dict[str, Any], bool]:
    healthy = len(ctx.router.healthy_replicas())
    total = len(ctx.router.replicas)
    ok = healthy > 0
    status = "healthy" if healthy == total and ok else ("degraded" if ok else "unhealthy")
    return (
        {
            "status": status,
            "model": ctx.served_model_name,
            "healthy_replicas": healthy,
            "total_replicas": total,
        },
        ok,
    )
