"""Static per-GPU memory model.

This model exists to *prune impossible candidates* and to seed engine memory settings. It is
deliberately conservative and is never the final word: launch validation and benchmarking decide.

Per GPU, an engine's memory budget is ``memory_fraction × total`` and must cover::

    weights (sharded) + activations + CUDA graphs + communication buffers + engine overhead + KV cache

The safety reserve (``headroom``) is the part of *free* memory ServePilot refuses to hand to the
engine. Memory already used by other processes is never treated as available.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from servepilot.constants import (
    ACTIVATION_FRACTION_OF_TOTAL,
    COMM_BUFFER_BYTES_PER_TP_RANK,
    CUDA_GRAPH_BYTES,
    DEFAULT_MEMORY_HEADROOM_FRACTION,
    ENGINE_OVERHEAD_BYTES,
    GIB,
    MAX_MEMORY_FRACTION,
    MIN_HEADROOM_BYTES,
    MIN_KV_SEQUENCES,
    MIN_MEMORY_FRACTION,
    WEIGHT_SHARD_OVERHEAD_FRACTION,
)
from servepilot.schemas.hardware import GPUDevice
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import KVEstimate, MemoryEstimate
from servepilot.schemas.workload import WorkloadProfile

# When the KV formula is unknown we still demand this much spare memory before calling a
# candidate viable; the engine will tell us the real capacity at launch.
UNKNOWN_KV_MIN_SPARE_BYTES = 2 * GIB


@dataclass(frozen=True)
class MemoryModelConfig:
    headroom_fraction: float = DEFAULT_MEMORY_HEADROOM_FRACTION
    min_headroom_bytes: int = MIN_HEADROOM_BYTES
    engine_overhead_bytes: int = ENGINE_OVERHEAD_BYTES
    cuda_graph_bytes: int = CUDA_GRAPH_BYTES
    comm_buffer_bytes_per_tp_rank: int = COMM_BUFFER_BYTES_PER_TP_RANK
    activation_fraction: float = ACTIVATION_FRACTION_OF_TOTAL
    weight_shard_overhead: float = WEIGHT_SHARD_OVERHEAD_FRACTION
    max_memory_fraction: float = MAX_MEMORY_FRACTION
    min_memory_fraction: float = MIN_MEMORY_FRACTION
    min_kv_sequences: int = MIN_KV_SEQUENCES


def safety_reserve_bytes(total_bytes: int, cfg: MemoryModelConfig) -> int:
    return int(max(cfg.headroom_fraction * total_bytes, cfg.min_headroom_bytes))


def compute_memory_fraction(total_bytes: int, free_bytes: int, cfg: MemoryModelConfig) -> float:
    """Largest engine memory fraction that leaves the safety reserve untouched.

    Engines interpret their fraction relative to *total* device memory, so the fraction is derived
    from what is actually free, rounded down to two decimals.
    """
    if total_bytes <= 0:
        return 0.0
    budget = free_bytes - safety_reserve_bytes(total_bytes, cfg)
    fraction = math.floor(budget / total_bytes * 100) / 100
    return max(0.0, min(cfg.max_memory_fraction, fraction))


def sharded_weight_bytes(weight_bytes: int, shards: int, cfg: MemoryModelConfig) -> int:
    if shards <= 1:
        return weight_bytes
    return int(weight_bytes / shards * (1.0 + cfg.weight_shard_overhead))


def estimate_memory(
    model: ModelProfile,
    *,
    gpus: Sequence[GPUDevice],
    tensor_parallel_size: int,
    kv: KVEstimate,
    context_length: int,
    workload: WorkloadProfile,
    cfg: MemoryModelConfig,
    pipeline_parallel_size: int = 1,
    memory_fraction: float | None = None,
) -> MemoryEstimate:
    """Estimate whether one replica of ``model`` fits on the GPUs of a single group."""
    if not gpus:
        raise ValueError("estimate_memory requires at least one GPU")
    total = min(g.total_memory_bytes for g in gpus)
    free = min(g.free_memory_bytes for g in gpus)
    fraction = (
        memory_fraction
        if memory_fraction is not None
        else compute_memory_fraction(total, free, cfg)
    )
    budget = int(fraction * total)
    reserve = safety_reserve_bytes(total, cfg)
    notes: list[str] = []
    shards = tensor_parallel_size * pipeline_parallel_size

    weights: int | None
    confidence: str
    if model.weight_bytes is None:
        weights = None
        confidence = "unknown"
        notes.append("model weight size unknown; viability can only be established by launching")
    else:
        weights = sharded_weight_bytes(model.weight_bytes, shards, cfg)
        confidence = "high" if model.weight_bytes_is_exact else "medium"
        if not model.weight_bytes_is_exact:
            notes.append(f"weight size is {model.weight_size_source}; treated as approximate")

    activations = int(cfg.activation_fraction * total)
    cuda_graphs = cfg.cuda_graph_bytes
    comm = cfg.comm_buffer_bytes_per_tp_rank if tensor_parallel_size > 1 else 0
    overhead = cfg.engine_overhead_bytes
    fixed = (weights or 0) + activations + cuda_graphs + comm + overhead
    kv_available = budget - fixed

    kv_per_gpu = kv.bytes_per_token_per_gpu
    if kv_per_gpu is not None and pipeline_parallel_size > 1:
        kv_per_gpu = max(1, kv_per_gpu // pipeline_parallel_size)

    est_tokens: int | None = None
    est_c50: int | None = None
    est_c95: int | None = None
    fits: bool
    shortfall = 0
    if weights is None:
        fits = kv_available >= UNKNOWN_KV_MIN_SPARE_BYTES
    elif kv_per_gpu:
        required_kv = kv_per_gpu * context_length * cfg.min_kv_sequences
        fits = kv_available >= required_kv
        if not fits:
            shortfall = required_kv - kv_available
        if kv_available > 0:
            est_tokens = kv_available // kv_per_gpu
            est_c50 = max(0, est_tokens // max(1, workload.p50_sequence_tokens))
            est_c95 = max(0, est_tokens // max(1, workload.p95_sequence_tokens))
        if kv.confidence in ("low", "medium") and confidence == "high":
            confidence = "medium"
    else:
        fits = kv_available >= UNKNOWN_KV_MIN_SPARE_BYTES
        if not fits:
            shortfall = UNKNOWN_KV_MIN_SPARE_BYTES - kv_available
        confidence = "low"
        notes.append(
            "KV-cache size unknown for this architecture; requiring 2 GiB spare before launch"
        )

    if free < total * 0.95:
        notes.append(
            f"{(total - free) / GIB:.1f} GiB already in use on the most occupied selected GPU; "
            "planning against free memory only"
        )
    if memory_fraction is not None and budget > free:
        # A forced fraction is honoured, but it cannot claim memory that is not free.
        notes.append(
            f"forced memory fraction {fraction:.2f} claims {format_bytes(budget)} per GPU but only "
            f"{format_bytes(free)} is free"
        )
        fits = False
        shortfall = max(shortfall, budget - free)
    if fraction < cfg.min_memory_fraction:
        notes.append(
            f"usable memory fraction {fraction:.2f} is below {cfg.min_memory_fraction:.2f}; GPU is heavily occupied"
        )
        fits = False
        shortfall = max(shortfall, int((cfg.min_memory_fraction - fraction) * total))
    if kv.explanation:
        notes.append(kv.explanation)

    return MemoryEstimate(
        device_total_bytes=total,
        device_free_bytes=free,
        memory_fraction=round(fraction, 2),
        engine_budget_bytes=budget,
        weights_bytes=weights,
        activations_bytes=activations,
        cuda_graph_bytes=cuda_graphs,
        communication_bytes=comm,
        engine_overhead_bytes=overhead,
        safety_reserve_bytes=reserve,
        kv_cache_bytes_available=max(kv_available, 0) if weights is not None else None,
        kv_bytes_per_token_per_gpu=kv_per_gpu,
        kv_confidence=kv.confidence,
        estimated_kv_tokens=est_tokens,
        estimated_max_concurrency_p50=est_c50,
        estimated_max_concurrency_p95=est_c95,
        fits=fits,
        confidence=confidence,  # type: ignore[arg-type]
        shortfall_bytes=max(0, shortfall),
        weight_source=model.weight_size_source,
        notes=notes,
    )


def minimum_tp_that_fits(
    model: ModelProfile,
    *,
    gpus: Sequence[GPUDevice],
    kv_for_tp: Callable[[int], KVEstimate],
    context_length: int,
    workload: WorkloadProfile,
    cfg: MemoryModelConfig,
    candidates: Sequence[int],
) -> int | None:
    """Smallest tensor-parallel size in ``candidates`` whose static estimate fits, or None."""
    for tp in sorted(candidates):
        est = estimate_memory(
            model,
            gpus=gpus,
            tensor_parallel_size=tp,
            kv=kv_for_tp(tp),
            context_length=context_length,
            workload=workload,
            cfg=cfg,
        )
        if est.fits:
            return tp
    return None


def format_bytes(n: int | None) -> str:
    if n is None:
        return "unknown"
    if abs(n) >= GIB:
        return f"{n / GIB:.1f} GiB"
    return f"{n / (1024 * 1024):.0f} MiB"
