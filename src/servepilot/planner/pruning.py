"""Static pruning rules. Each returns a human-readable reason or ``None`` when the check passes."""

from __future__ import annotations

from collections.abc import Iterable

from servepilot.planner.memory import format_bytes
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidatePlan, MemoryEstimate


def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def check_tp_divisibility(model: ModelProfile, tp: int) -> str | None:
    """Plain TP only needs the attention heads to split; expert counts are checked by
    :func:`check_ep_divisibility` when an expert-parallel variant is proposed."""
    heads = model.num_attention_heads
    if heads is not None and heads % tp != 0:
        return f"{heads} attention heads are not divisible by TP={tp}"
    return None


def check_ep_divisibility(model: ModelProfile, ep: int) -> str | None:
    if model.num_experts is not None and model.num_experts % ep != 0:
        return f"{model.num_experts} experts are not divisible by EP={ep}"
    return None


def check_context_length(
    model: ModelProfile, context_length: int, allow_override: bool
) -> str | None:
    limit = model.max_position_embeddings
    if limit is not None and context_length > limit and not allow_override:
        return (
            f"requested context {context_length} exceeds the model's max_position_embeddings ({limit}); "
            "pass --allow-context-override to force it"
        )
    return None


def check_memory(estimate: MemoryEstimate, tolerance_fraction: float = 0.0) -> str | None:
    if estimate.fits:
        return None
    budget = estimate.engine_budget_bytes
    if estimate.shortfall_bytes <= tolerance_fraction * budget:
        return None
    if budget > estimate.device_free_bytes:
        return (
            f"memory fraction {estimate.memory_fraction:.2f} claims {format_bytes(budget)} per GPU "
            f"but only {format_bytes(estimate.device_free_bytes)} is free"
        )
    parts = [
        f"estimated per-GPU requirement exceeds the safe budget by {format_bytes(estimate.shortfall_bytes)}",
        f"(weights {format_bytes(estimate.weights_bytes)} + overheads {format_bytes(estimate.fixed_bytes - (estimate.weights_bytes or 0))}",
        f"vs budget {format_bytes(budget)} at memory fraction {estimate.memory_fraction:.2f})",
    ]
    return " ".join(parts)


def dedupe(plans: Iterable[CandidatePlan]) -> tuple[list[CandidatePlan], list[CandidatePlan]]:
    """Split plans into (unique, duplicates) by structural key, keeping first occurrences."""
    seen: set[str] = set()
    unique: list[CandidatePlan] = []
    dupes: list[CandidatePlan] = []
    for p in plans:
        key = p.structural_key()
        if key in seen:
            dupes.append(p)
        else:
            seen.add(key)
            unique.append(p)
    return unique, dupes
