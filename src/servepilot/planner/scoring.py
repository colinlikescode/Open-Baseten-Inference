"""Objective scoring, SLO filtering, plateau detection and Pareto frontier.

Formulas (documented in docs/benchmarking.md):

* **throughput** – ``score = output_tokens_per_second × error_factor``
* **latency**    – ``score = (0.5·L + 0.25·T + 0.25·P) × error_factor`` where ``L``, ``T``, ``P``
  are ``best/candidate`` ratios for p95 end-to-end latency, p95 TTFT and p95 TPOT (a metric that
  no candidate reports is dropped and the weights renormalised).
* **balanced**   – ``score = sqrt(throughput_norm × latency_norm) × error_factor`` with
  ``throughput_norm = tps / best_tps`` and ``latency_norm = best_p95 / candidate_p95``.
* ``error_factor = max(0, 1 − ERROR_RATE_PENALTY_WEIGHT × error_rate)``; results whose error
  rate exceeds ``MAX_ACCEPTABLE_ERROR_RATE`` are invalid and never selected while a valid result
  exists.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from servepilot.constants import (
    ERROR_RATE_PENALTY_WEIGHT,
    MAX_ACCEPTABLE_ERROR_RATE,
    PLATEAU_LATENCY_RISE_THRESHOLD,
    PLATEAU_MIN_THROUGHPUT_GAIN,
    SWEEP_THROUGHPUT_DROP_STOP,
)
from servepilot.schemas.benchmark import BenchmarkResult, ParetoPoint
from servepilot.schemas.workload import LatencyConstraints, Objective


@dataclass(frozen=True)
class ScoringConfig:
    max_error_rate: float = MAX_ACCEPTABLE_ERROR_RATE
    error_penalty_weight: float = ERROR_RATE_PENALTY_WEIGHT
    plateau_min_gain: float = PLATEAU_MIN_THROUGHPUT_GAIN
    plateau_latency_rise: float = PLATEAU_LATENCY_RISE_THRESHOLD
    throughput_drop_stop: float = SWEEP_THROUGHPUT_DROP_STOP


@dataclass
class ScoredResult:
    result: BenchmarkResult
    score: float
    valid: bool
    slo_satisfied: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return self.valid and self.slo_satisfied


def slo_violations(result: BenchmarkResult, slo: LatencyConstraints | None) -> list[str]:
    if slo is None or slo.is_empty:
        return []
    out: list[str] = []
    if slo.max_p95_ttft_ms is not None:
        if result.ttft_p95_ms is None:
            out.append("p95 TTFT unavailable; cannot verify the requested SLO")
        elif result.ttft_p95_ms > slo.max_p95_ttft_ms:
            out.append(f"p95 TTFT {result.ttft_p95_ms:.0f} ms > {slo.max_p95_ttft_ms:.0f} ms")
    if slo.max_p95_latency_ms is not None and result.latency_p95_ms > slo.max_p95_latency_ms:
        out.append(f"p95 latency {result.latency_p95_ms:.0f} ms > {slo.max_p95_latency_ms:.0f} ms")
    if slo.max_p95_tpot_ms is not None:
        if result.tpot_p95_ms is None:
            out.append("p95 TPOT unavailable; cannot verify the requested SLO")
        elif result.tpot_p95_ms > slo.max_p95_tpot_ms:
            out.append(f"p95 TPOT {result.tpot_p95_ms:.1f} ms > {slo.max_p95_tpot_ms:.1f} ms")
    return out


def error_factor(result: BenchmarkResult, cfg: ScoringConfig) -> float:
    return max(0.0, 1.0 - cfg.error_penalty_weight * result.error_rate)


def is_valid(result: BenchmarkResult, cfg: ScoringConfig) -> tuple[bool, str | None]:
    if result.successful_requests == 0:
        return False, "no request succeeded"
    if result.error_rate > cfg.max_error_rate:
        return False, f"error rate {result.error_rate:.1%} exceeds {cfg.max_error_rate:.1%}"
    return True, None


def _ratio(best: float | None, value: float | None) -> float | None:
    if best is None or value is None or value <= 0:
        return None
    return best / value


def score_results(
    results: Sequence[BenchmarkResult],
    objective: Objective,
    slo: LatencyConstraints | None = None,
    cfg: ScoringConfig | None = None,
) -> list[ScoredResult]:
    """Score a set of results relative to each other (normalisation uses the *valid* set)."""
    cfg = cfg or ScoringConfig()
    scored: list[ScoredResult] = []
    validity = [is_valid(r, cfg) for r in results]
    valid_results = [r for r, (ok, _) in zip(results, validity, strict=True) if ok]

    best_tps = max((r.output_tokens_per_second for r in valid_results), default=None)
    best_lat = min((r.latency_p95_ms for r in valid_results if r.latency_p95_ms > 0), default=None)
    best_ttft = min(
        (r.ttft_p95_ms for r in valid_results if r.ttft_p95_ms is not None), default=None
    )
    best_tpot = min(
        (r.tpot_p95_ms for r in valid_results if r.tpot_p95_ms is not None), default=None
    )

    for r, (ok, why) in zip(results, validity, strict=True):
        reasons: list[str] = []
        if why:
            reasons.append(why)
        violations = slo_violations(r, slo)
        reasons.extend(violations)
        factor = error_factor(r, cfg)
        if objective == Objective.THROUGHPUT:
            score = r.output_tokens_per_second * factor
        elif objective == Objective.LATENCY:
            parts: list[tuple[float, float]] = []
            for weight, ratio in (
                (0.5, _ratio(best_lat, r.latency_p95_ms)),
                (0.25, _ratio(best_ttft, r.ttft_p95_ms)),
                (0.25, _ratio(best_tpot, r.tpot_p95_ms)),
            ):
                if ratio is not None:
                    parts.append((weight, ratio))
            total_w = sum(w for w, _ in parts)
            score = (sum(w * v for w, v in parts) / total_w if total_w else 0.0) * factor
        else:
            t_norm = (r.output_tokens_per_second / best_tps) if best_tps else 0.0
            l_norm = _ratio(best_lat, r.latency_p95_ms) or 0.0
            score = math.sqrt(max(t_norm, 0.0) * max(l_norm, 0.0)) * factor
        scored.append(
            ScoredResult(
                result=r, score=score, valid=ok, slo_satisfied=not violations, reasons=reasons
            )
        )
    return scored


@dataclass
class Selection:
    winner: ScoredResult
    slo_satisfied: bool
    notes: list[str] = field(default_factory=list)


def select_best(scored: Sequence[ScoredResult]) -> Selection | None:
    """Pick the winner: eligible results first, then valid-but-SLO-violating, else None."""
    if not scored:
        return None
    eligible = [s for s in scored if s.eligible]
    if eligible:
        return Selection(winner=max(eligible, key=lambda s: s.score), slo_satisfied=True)
    valid = [s for s in scored if s.valid]
    if valid:
        best = max(valid, key=lambda s: s.score)
        return Selection(
            winner=best,
            slo_satisfied=False,
            notes=[
                "No tested configuration satisfied the requested SLO; reporting the best observed."
            ],
        )
    return None


# ---------------------------------------------------------------------------
# Concurrency sweep decisions
# ---------------------------------------------------------------------------
@dataclass
class SweepDecision:
    continue_sweep: bool
    reason: str


def should_continue_sweep(
    history: Sequence[BenchmarkResult],
    slo: LatencyConstraints | None,
    cfg: ScoringConfig | None = None,
) -> SweepDecision:
    """Decide whether raising concurrency further is worthwhile given results so far."""
    cfg = cfg or ScoringConfig()
    if not history:
        return SweepDecision(True, "no data yet")
    last = history[-1]
    ok, why = is_valid(last, cfg)
    if not ok:
        return SweepDecision(
            False, f"failures appeared at concurrency {last.spec.concurrency}: {why}"
        )
    if slo_violations(last, slo):
        return SweepDecision(False, f"SLO violated at concurrency {last.spec.concurrency}")
    if len(history) < 2:
        return SweepDecision(True, "need at least two points")
    best_tps = max(r.output_tokens_per_second for r in history)
    if last.output_tokens_per_second < best_tps * (1 - cfg.throughput_drop_stop):
        return SweepDecision(
            False,
            f"throughput dropped {1 - last.output_tokens_per_second / best_tps:.1%} from the best point",
        )
    prev = history[-2]
    gain = (last.output_tokens_per_second - prev.output_tokens_per_second) / max(
        prev.output_tokens_per_second, 1e-9
    )
    lat_rise = (last.latency_p95_ms - prev.latency_p95_ms) / max(prev.latency_p95_ms, 1e-9)
    if gain < cfg.plateau_min_gain and lat_rise > cfg.plateau_latency_rise:
        return SweepDecision(
            False, f"throughput plateaued (+{gain:.1%}) while p95 latency rose {lat_rise:.0%}"
        )
    if gain < cfg.plateau_min_gain and len(history) >= 3:
        prev_gain = (prev.output_tokens_per_second - history[-3].output_tokens_per_second) / max(
            history[-3].output_tokens_per_second, 1e-9
        )
        if prev_gain < cfg.plateau_min_gain:
            return SweepDecision(False, "throughput plateaued for two consecutive steps")
    return SweepDecision(True, f"throughput still improving (+{gain:.1%})")


@dataclass
class ConcurrencyChoice:
    result: BenchmarkResult
    reason: str


def choose_concurrency(
    history: Sequence[BenchmarkResult],
    objective: Objective,
    slo: LatencyConstraints | None,
    cfg: ScoringConfig | None = None,
) -> ConcurrencyChoice | None:
    """Apply the plateau rule to a concurrency sweep and pick the operating point.

    Throughput/balanced: walk upward; a step that gains less than ``plateau_min_gain`` throughput
    while raising p95 latency more than ``plateau_latency_rise`` does not count as an
    improvement, so the lower concurrency wins. Latency objective: highest scoring point that
    satisfies the SLO (or the lowest-latency point when no SLO is given).
    """
    cfg = cfg or ScoringConfig()
    points = sorted((r for r in history if is_valid(r, cfg)[0]), key=lambda r: r.spec.concurrency)
    if not points:
        return None
    eligible = [r for r in points if not slo_violations(r, slo)]
    pool = eligible or points
    note = "" if eligible else " (no point satisfied the SLO; using best observed)"

    if objective == Objective.LATENCY:
        best = min(pool, key=lambda r: r.latency_p95_ms)
        if eligible and slo is not None and not slo.is_empty:
            # Under an SLO, the useful operating point is the highest concurrency that still meets it.
            best = max(eligible, key=lambda r: r.spec.concurrency)
            return ConcurrencyChoice(
                best, f"highest concurrency ({best.spec.concurrency}) that satisfies the SLO"
            )
        return ConcurrencyChoice(
            best, f"lowest p95 latency at concurrency {best.spec.concurrency}{note}"
        )

    current = pool[0]
    reason = f"concurrency {current.spec.concurrency} is the only valid point"
    for nxt in pool[1:]:
        gain = (nxt.output_tokens_per_second - current.output_tokens_per_second) / max(
            current.output_tokens_per_second, 1e-9
        )
        lat_rise = (nxt.latency_p95_ms - current.latency_p95_ms) / max(current.latency_p95_ms, 1e-9)
        if gain <= 0:
            reason = f"concurrency {nxt.spec.concurrency} did not improve throughput over {current.spec.concurrency}"
            continue
        if gain < cfg.plateau_min_gain and lat_rise > cfg.plateau_latency_rise:
            reason = (
                f"at {nxt.spec.concurrency} concurrent requests throughput improved {gain:.1%} while p95 latency "
                f"rose {lat_rise:.0%}; {current.spec.concurrency} is preferred under the plateau rule"
            )
            continue
        current = nxt
        reason = f"concurrency {current.spec.concurrency} had the highest throughput under the plateau rule"
    return ConcurrencyChoice(current, reason + note)


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------
def pareto_front(results: Sequence[BenchmarkResult]) -> list[ParetoPoint]:
    """Mark the non-dominated (throughput ↑, p95 latency ↓) points; returns all points sorted."""
    points = [
        ParetoPoint(
            candidate_id=r.candidate_id,
            concurrency=r.spec.concurrency,
            output_tokens_per_second=r.output_tokens_per_second,
            latency_p95_ms=r.latency_p95_ms,
        )
        for r in results
        if r.successful_requests > 0
    ]
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            better_or_equal = (
                q.output_tokens_per_second >= p.output_tokens_per_second
                and q.latency_p95_ms <= p.latency_p95_ms
            )
            strictly = (
                q.output_tokens_per_second > p.output_tokens_per_second
                or q.latency_p95_ms < p.latency_p95_ms
            )
            if better_or_equal and strictly:
                dominated = True
                break
        p.on_front = not dominated
    return sorted(points, key=lambda p: (p.latency_p95_ms, -p.output_tokens_per_second))
