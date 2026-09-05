"""Aggregate per-request measurements into a :class:`BenchmarkResult`.

Definitions (see docs/benchmarking.md):

* ``duration_seconds`` – wall clock from the first request start to the last completion.
* ``output_tokens_per_second`` – successful output tokens / duration (aggregate work, never
  derived from ``1 / mean latency``).
* ``request_throughput`` – successful requests / duration.
* Percentiles are linearly interpolated over successful requests only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from servepilot.schemas.benchmark import (
    BenchmarkResult,
    BenchmarkSpec,
    GPUBenchmarkMetrics,
    RequestBenchmarkResult,
)
from servepilot.schemas.hardware import GPUSample


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Linear-interpolated percentile (``pct`` in 0..100). None for an empty sequence."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def aggregate_gpu_samples(samples: Sequence[GPUSample]) -> GPUBenchmarkMetrics:
    if not samples:
        return GPUBenchmarkMetrics()
    utils = [s.utilization_percent for s in samples if s.utilization_percent is not None]
    mems = [s.memory_used_bytes for s in samples if s.memory_used_bytes is not None]
    power = [s.power_watts for s in samples if s.power_watts is not None]
    return GPUBenchmarkMetrics(
        mean_gpu_utilization=(sum(utils) / len(utils)) if utils else None,
        peak_gpu_utilization=max(utils) if utils else None,
        peak_memory_bytes=max(mems) if mems else None,
        mean_power_watts=(sum(power) / len(power)) if power else None,
        sample_count=len(samples),
    )


def aggregate(
    candidate_id: str,
    spec: BenchmarkSpec,
    results: Sequence[RequestBenchmarkResult],
    *,
    gpu_samples: Sequence[GPUSample] = (),
    wall_duration_seconds: float | None = None,
) -> BenchmarkResult:
    total = len(results)
    ok = [r for r in results if r.success]
    failed = total - len(ok)

    if wall_duration_seconds is not None:
        duration = wall_duration_seconds
    elif results:
        start = min(r.started_at for r in results)
        end = max((r.completed_at or r.started_at) for r in results)
        duration = max(end - start, 1e-6)
    else:
        duration = 1e-6

    in_tokens = sum(r.input_tokens for r in ok)
    out_tokens = sum(r.output_tokens for r in ok)
    latencies = [r.e2e_latency_ms for r in ok if r.e2e_latency_ms is not None]
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms is not None]
    errors = [r.error for r in results if r.error]

    def p(values: list[float], pct: float) -> float | None:
        return percentile(values, pct)

    return BenchmarkResult(
        candidate_id=candidate_id,
        spec=spec,
        total_requests=total,
        successful_requests=len(ok),
        failed_requests=failed,
        duration_seconds=duration,
        request_throughput=len(ok) / duration,
        input_tokens_per_second=in_tokens / duration,
        output_tokens_per_second=out_tokens / duration,
        total_tokens_per_second=(in_tokens + out_tokens) / duration,
        ttft_p50_ms=p(ttfts, 50),
        ttft_p95_ms=p(ttfts, 95),
        ttft_p99_ms=p(ttfts, 99),
        tpot_p50_ms=p(tpots, 50),
        tpot_p95_ms=p(tpots, 95),
        tpot_p99_ms=p(tpots, 99),
        latency_p50_ms=p(latencies, 50) or 0.0,
        latency_p95_ms=p(latencies, 95) or 0.0,
        latency_p99_ms=p(latencies, 99) or 0.0,
        error_rate=(failed / total) if total else 1.0,
        gpu_metrics=aggregate_gpu_samples(gpu_samples),
        errors_sample=sorted(set(errors))[:5],
    )
