"""Objective scoring, SLO filtering, plateau detection and Pareto frontier."""

from __future__ import annotations

import pytest

from servepilot.benchmark.metrics import percentile
from servepilot.planner.scoring import (
    ScoringConfig,
    choose_concurrency,
    pareto_front,
    score_results,
    select_best,
    should_continue_sweep,
    slo_violations,
)
from servepilot.schemas.benchmark import BenchmarkResult, BenchmarkSpec
from servepilot.schemas.workload import LatencyConstraints, Objective


def result(
    cid: str,
    tps: float,
    p95: float,
    *,
    concurrency: int = 16,
    ttft: float | None = 100.0,
    tpot: float | None = 10.0,
    error_rate: float = 0.0,
    total: int = 100,
    label: str | None = None,
) -> BenchmarkResult:
    failed = round(total * error_rate)
    return BenchmarkResult(
        candidate_id=cid,
        spec=BenchmarkSpec(
            concurrency=concurrency,
            num_requests=total,
            seed=1,
            input_tokens_p50=1,
            input_tokens_p95=1,
            output_tokens_p50=1,
            output_tokens_p95=1,
            label=label,
        ),
        total_requests=total,
        successful_requests=total - failed,
        failed_requests=failed,
        duration_seconds=10.0,
        request_throughput=(total - failed) / 10.0,
        input_tokens_per_second=tps,
        output_tokens_per_second=tps,
        total_tokens_per_second=2 * tps,
        ttft_p50_ms=ttft,
        ttft_p95_ms=ttft,
        ttft_p99_ms=ttft,
        tpot_p50_ms=tpot,
        tpot_p95_ms=tpot,
        tpot_p99_ms=tpot,
        latency_p50_ms=p95 * 0.7,
        latency_p95_ms=p95,
        latency_p99_ms=p95 * 1.2,
        error_rate=failed / total,
    )


class TestObjectives:
    def test_throughput_prefers_tokens_per_second(self) -> None:
        scored = score_results(
            [result("a", 1000, 900), result("b", 800, 500), result("c", 500, 300)],
            Objective.THROUGHPUT,
        )
        best = select_best(scored)
        assert best is not None and best.winner.result.candidate_id == "a"

    def test_latency_prefers_low_p95(self) -> None:
        scored = score_results(
            [result("a", 1000, 900), result("b", 800, 500), result("c", 500, 300)],
            Objective.LATENCY,
        )
        best = select_best(scored)
        assert best is not None and best.winner.result.candidate_id == "c"

    def test_latency_uses_ttft_and_tpot_when_present(self) -> None:
        # Same e2e latency; b has much better TTFT/TPOT.
        scored = score_results(
            [result("a", 500, 500, ttft=400, tpot=20), result("b", 500, 500, ttft=100, tpot=5)],
            Objective.LATENCY,
        )
        assert scored[1].score > scored[0].score
        # Missing sub-metrics are dropped and weights renormalised (no crash, still comparable).
        scored2 = score_results(
            [
                result("a", 500, 500, ttft=None, tpot=None),
                result("b", 500, 600, ttft=None, tpot=None),
            ],
            Objective.LATENCY,
        )
        assert scored2[0].score == pytest.approx(1.0) and scored2[1].score < 1.0

    def test_balanced_is_geometric_mean(self) -> None:
        scored = score_results(
            [result("fast", 1000, 1000), result("lowlat", 500, 400), result("mid", 900, 500)],
            Objective.BALANCED,
        )
        by_id = {s.result.candidate_id: s.score for s in scored}
        # mid: sqrt(0.9 * 400/500) = sqrt(0.72) = 0.8485
        assert by_id["mid"] == pytest.approx((0.9 * 0.8) ** 0.5)
        assert by_id["fast"] == pytest.approx((1.0 * 0.4) ** 0.5)
        assert max(by_id, key=by_id.get) == "mid"  # type: ignore[arg-type]

    def test_error_penalty_and_invalidity(self) -> None:
        cfg = ScoringConfig()
        scored = score_results(
            [
                result("clean", 900, 500),
                result("flaky", 1000, 500, error_rate=0.005, total=200),
                result("broken", 5000, 100, error_rate=0.2),
            ],
            Objective.THROUGHPUT,
            cfg=cfg,
        )
        by_id = {s.result.candidate_id: s for s in scored}
        assert by_id["broken"].valid is False and "error rate" in by_id["broken"].reasons[0]
        assert by_id["flaky"].valid and by_id["flaky"].score == pytest.approx(
            1000 * (1 - 5.0 * 0.005)
        )
        best = select_best(scored)
        assert best is not None and best.winner.result.candidate_id == "flaky"
        all_broken = score_results([result("x", 100, 100, error_rate=0.5)], Objective.THROUGHPUT)
        assert select_best(all_broken) is None

    def test_no_successes_is_invalid(self) -> None:
        scored = score_results([result("dead", 0, 0, error_rate=1.0)], Objective.THROUGHPUT)
        assert not scored[0].valid


class TestSLO:
    slo = LatencyConstraints(max_p95_latency_ms=600, max_p95_ttft_ms=150)

    @pytest.mark.parametrize("metric", ["ttft", "tpot"])
    def test_missing_required_metric_cannot_satisfy_slo(self, metric: str) -> None:
        slo = LatencyConstraints.model_validate({f"max_p95_{metric}_ms": 150})
        missing = result("unmeasured", 2000, 500, ttft=None, tpot=None)
        measured = result("measured", 1000, 500)
        violations = slo_violations(missing, slo)
        assert len(violations) == 1 and metric.upper() in violations[0]
        scored = score_results([missing, measured], Objective.THROUGHPUT, slo)
        best = select_best(scored)
        assert best is not None and best.winner.result.candidate_id == "measured"
        fallback = select_best(scored[:1])
        assert fallback is not None and not fallback.slo_satisfied
        assert not should_continue_sweep([missing], slo).continue_sweep

    def test_missing_optional_metrics_do_not_violate_slo(self) -> None:
        missing = result("unmeasured", 2000, 500, ttft=None, tpot=None)
        assert slo_violations(missing, LatencyConstraints(max_p95_latency_ms=600)) == []

    def test_violations_listed(self) -> None:
        v = slo_violations(result("a", 1000, 900, ttft=200), self.slo)
        assert (
            len(v) == 2 and any("p95 latency" in x for x in v) and any("p95 TTFT" in x for x in v)
        )
        assert slo_violations(result("b", 1000, 500), self.slo) == []
        assert slo_violations(result("c", 1000, 5000), None) == []

    def test_slo_rejection_prefers_compliant_candidate(self) -> None:
        scored = score_results(
            [result("fast", 2000, 900), result("ok", 1000, 500)], Objective.THROUGHPUT, self.slo
        )
        best = select_best(scored)
        assert best is not None and best.winner.result.candidate_id == "ok" and best.slo_satisfied

    def test_no_candidate_meets_slo(self) -> None:
        scored = score_results(
            [result("a", 2000, 900), result("b", 1000, 800)], Objective.THROUGHPUT, self.slo
        )
        best = select_best(scored)
        assert (
            best is not None and not best.slo_satisfied and best.winner.result.candidate_id == "a"
        )
        assert "No tested configuration satisfied" in best.notes[0]


class TestSweep:
    def test_stop_conditions(self) -> None:
        cfg = ScoringConfig()
        hist = [result("a", 1000, 500, concurrency=8), result("a", 1900, 520, concurrency=16)]
        assert should_continue_sweep(hist, None, cfg).continue_sweep
        plateau = [*hist, result("a", 1910, 700, concurrency=32)]
        d = should_continue_sweep(plateau, None, cfg)
        assert not d.continue_sweep and "plateaued" in d.reason
        drop = [*hist, result("a", 1500, 520, concurrency=32)]
        assert not should_continue_sweep(drop, None, cfg).continue_sweep
        errors = [*hist, result("a", 2500, 520, concurrency=32, error_rate=0.1)]
        assert "failures" in should_continue_sweep(errors, None, cfg).reason
        slo = [*hist, result("a", 2500, 5000, concurrency=32)]
        assert (
            "SLO"
            in should_continue_sweep(slo, LatencyConstraints(max_p95_latency_ms=1000), cfg).reason
        )
        flat_twice = [
            *hist,
            result("a", 1905, 530, concurrency=32),
            result("a", 1908, 540, concurrency=64),
        ]
        assert "two consecutive" in should_continue_sweep(flat_twice, None, cfg).reason
        assert should_continue_sweep([], None, cfg).continue_sweep

    def test_plateau_rule_prefers_lower_concurrency(self) -> None:
        hist = [
            result("a", 11_420, 890, concurrency=32),
            result("a", 18_100, 1_480, concurrency=64),
            result("a", 20_040, 2_310, concurrency=96),
            result("a", 20_190, 3_780, concurrency=128),  # +0.7% throughput, +64% latency
        ]
        choice = choose_concurrency(hist, Objective.THROUGHPUT, None)
        assert choice is not None and choice.result.spec.concurrency == 96
        assert "plateau rule" in choice.reason

    def test_highest_throughput_wins_when_gains_are_real(self) -> None:
        hist = [
            result("a", 1000, 500, concurrency=8),
            result("a", 1900, 520, concurrency=16),
            result("a", 3500, 600, concurrency=32),
        ]
        choice = choose_concurrency(hist, Objective.THROUGHPUT, None)
        assert choice is not None and choice.result.spec.concurrency == 32

    def test_latency_objective_choices(self) -> None:
        hist = [
            result("a", 1000, 300, concurrency=8),
            result("a", 1900, 450, concurrency=16),
            result("a", 3500, 900, concurrency=32),
        ]
        no_slo = choose_concurrency(hist, Objective.LATENCY, None)
        assert no_slo is not None and no_slo.result.spec.concurrency == 8
        with_slo = choose_concurrency(
            hist, Objective.LATENCY, LatencyConstraints(max_p95_latency_ms=500)
        )
        assert with_slo is not None and with_slo.result.spec.concurrency == 16
        none_meet = choose_concurrency(
            hist, Objective.LATENCY, LatencyConstraints(max_p95_latency_ms=100)
        )
        assert none_meet is not None and "no point satisfied" in none_meet.reason

    def test_invalid_points_ignored(self) -> None:
        hist = [
            result("a", 1000, 500, concurrency=8),
            result("a", 5000, 600, concurrency=64, error_rate=0.3),
        ]
        choice = choose_concurrency(hist, Objective.THROUGHPUT, None)
        assert choice is not None and choice.result.spec.concurrency == 8
        assert (
            choose_concurrency([result("a", 0, 0, error_rate=1.0)], Objective.THROUGHPUT, None)
            is None
        )


class TestPareto:
    def test_front(self) -> None:
        pts = pareto_front(
            [
                result("a", 11_420, 890, concurrency=32),
                result("a", 18_100, 1_480, concurrency=64),
                result("a", 17_000, 1_600, concurrency=80),  # dominated by 64
                result("a", 20_190, 3_780, concurrency=128),
                result("b", 0, 0, error_rate=1.0),  # no successes: excluded
            ]
        )
        assert len(pts) == 4
        front = [(p.concurrency, p.on_front) for p in pts]
        assert (80, False) in front and all(on for c, on in front if c != 80)
        assert pts[0].latency_p95_ms <= pts[-1].latency_p95_ms


class TestPercentile:
    def test_percentile(self) -> None:
        assert percentile([], 50) is None
        assert percentile([5.0], 99) == 5.0
        assert percentile([1, 2, 3, 4, 5], 50) == 3
        assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
        assert percentile([1, 2, 3, 4], 100) == 4
