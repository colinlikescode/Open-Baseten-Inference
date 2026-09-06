"""Staged tuner against scripted candidate evaluators (no processes)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from servepilot.benchmark.evaluator import CandidateLaunchFailed
from servepilot.benchmark.tuner import Tuner, TuningSettings
from servepilot.exceptions import NoViablePlanError
from servepilot.planner.candidates import generate_candidates
from servepilot.schemas.benchmark import (
    BenchmarkResult,
    BenchmarkSpec,
    CandidateFailure,
    FailureType,
)
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidatePlan, PlanningResult
from servepilot.schemas.workload import (
    LatencyConstraints,
    Objective,
    WorkloadProfile,
    workload_from_preset,
)
from servepilot.testing import fake_hardware as fh
from servepilot.testing.fake_engine import FakeEngine

# (throughput tok/s, p95 latency ms, p95 ttft ms, error rate) as a function of concurrency
PerfModel = Callable[[int], tuple[float, float, float, float]]


def saturating(peak_tps: float, capacity: int, base_latency: float = 400.0) -> PerfModel:
    """Throughput rises linearly to ``capacity`` then plateaus; latency grows past capacity."""

    def model(c: int) -> tuple[float, float, float, float]:
        util = min(c, capacity) / capacity
        tps = peak_tps * util
        latency = base_latency * (1 + max(0, c - capacity) / capacity * 2)
        ttft = 50 + max(0, c - capacity) * 5
        return tps, latency, ttft, 0.0

    return model


@dataclass
class ScriptedSession:
    plan: CandidatePlan
    perf: PerfModel
    evaluator: ScriptedEvaluator
    launch_seconds: float = 1.0
    engine_version: str | None = "0.0.1-fake"
    runtime_metadata: dict[str, object] = field(default_factory=dict)
    closed: bool = False

    async def benchmark(self, spec: BenchmarkSpec) -> BenchmarkResult:
        assert not self.closed
        self.evaluator.benchmarks.append((self.plan.id, spec.concurrency, spec.label or ""))
        tps, p95, ttft, err = self.perf(spec.concurrency)
        failed = round(spec.num_requests * err)
        ok = spec.num_requests - failed
        return BenchmarkResult(
            candidate_id=self.plan.id,
            spec=spec,
            total_requests=spec.num_requests,
            successful_requests=ok,
            failed_requests=failed,
            duration_seconds=10.0,
            request_throughput=ok / 10.0,
            input_tokens_per_second=tps,
            output_tokens_per_second=tps,
            total_tokens_per_second=2 * tps,
            ttft_p50_ms=ttft * 0.8,
            ttft_p95_ms=ttft,
            ttft_p99_ms=ttft * 1.2,
            tpot_p50_ms=8.0,
            tpot_p95_ms=10.0,
            tpot_p99_ms=12.0,
            latency_p50_ms=p95 * 0.7,
            latency_p95_ms=p95,
            latency_p99_ms=p95 * 1.2,
            error_rate=failed / spec.num_requests,
        )

    async def close(self) -> None:
        self.closed = True
        self.evaluator.open_sessions.remove(self)


class ScriptedEvaluator:
    """Maps ``tp{N}`` (or plan id) → performance model or startup failure."""

    def __init__(
        self, perf: dict[str, PerfModel], failures: dict[str, CandidateFailure] | None = None
    ) -> None:
        self.perf = perf
        self.failures = failures or {}
        self.opens: list[str] = []
        self.benchmarks: list[tuple[str, int, str]] = []
        self.open_sessions: list[ScriptedSession] = []
        self.max_concurrent_sessions = 0

    def _key(self, plan: CandidatePlan) -> str:
        for key in (plan.id, f"tp{plan.tensor_parallel_size}"):
            if key in self.perf or key in self.failures:
                return key
        raise KeyError(plan.id)

    async def open(self, plan: CandidatePlan) -> ScriptedSession:
        key = self._key(plan)
        self.opens.append(plan.id)
        if key in self.failures:
            raise CandidateLaunchFailed(self.failures[key])
        session = ScriptedSession(plan=plan, perf=self.perf[key], evaluator=self)
        self.open_sessions.append(session)
        self.max_concurrent_sessions = max(self.max_concurrent_sessions, len(self.open_sessions))
        return session


def planning(model: ModelProfile, workload: WorkloadProfile, gpus: int = 4) -> PlanningResult:
    return generate_candidates(fh.h100(gpus), model, workload, [FakeEngine()])


SETTINGS = TuningSettings(
    stage_a_requests=8, sweep_requests=8, final_multiplier=2, sweep_start=4, memory_tuning=False
)


class TestSelection:
    async def test_throughput_picks_fastest_topology(self, dense_8b: ModelProfile) -> None:
        wl = workload_from_preset("chat")
        evaluator = ScriptedEvaluator(
            {"tp1": saturating(1000, 64), "tp2": saturating(800, 64), "tp4": saturating(500, 64)}
        )
        outcome = await Tuner(evaluator, wl, SETTINGS).tune(planning(dense_8b, wl))
        winner = outcome.winner
        assert winner.plan.tensor_parallel_size == 1 and winner.plan.replica_count == 4
        assert winner.benchmarked and winner.source == "tuned"
        assert winner.final_result is not None and winner.final_result.spec.label == "final"
        assert winner.plan.max_concurrency == winner.final_result.spec.concurrency
        assert evaluator.max_concurrent_sessions == 1, "candidates must be benchmarked in isolation"
        assert not evaluator.open_sessions, "all sessions must be closed"
        assert any("peaked at" in line for line in winner.rationale)
        assert winner.pareto_front and any(p.on_front for p in winner.pareto_front)
        stages = {e.stage for e in outcome.evaluations}
        assert {"structural", "concurrency", "final"} <= stages

    async def test_latency_objective_changes_selection(self, dense_8b: ModelProfile) -> None:
        wl = workload_from_preset("chat", objective=Objective.LATENCY, expected_concurrency=16)

        def fast_but_slow_latency(c: int) -> tuple[float, float, float, float]:
            return 1000.0, 900.0, 300.0, 0.0

        def low_latency(c: int) -> tuple[float, float, float, float]:
            return 500.0, 300.0, 80.0, 0.0

        evaluator = ScriptedEvaluator(
            {
                "tp1": fast_but_slow_latency,
                "tp2": saturating(800, 64, base_latency=600),
                "tp4": low_latency,
            }
        )
        outcome = await Tuner(evaluator, wl, SETTINGS).tune(planning(dense_8b, wl))
        assert outcome.winner.plan.tensor_parallel_size == 4
        # Latency objective benchmarks at the expected concurrency, never only at 1.
        assert all(
            c == 16
            for _, c, label in evaluator.benchmarks
            if label in ("stage-a", "latency", "final")
        )

    async def test_latency_without_expected_concurrency_reports_assumption(
        self, dense_8b: ModelProfile
    ) -> None:
        wl = workload_from_preset("chat", objective=Objective.LATENCY)
        evaluator = ScriptedEvaluator(
            {"tp1": saturating(1000, 64), "tp2": saturating(800, 64), "tp4": saturating(500, 64)}
        )
        outcome = await Tuner(evaluator, wl, SETTINGS).tune(planning(dense_8b, wl))
        assert any("No expected concurrency" in n for n in outcome.notes)

    async def test_balanced(self, dense_8b: ModelProfile) -> None:
        wl = workload_from_preset("chat", objective=Objective.BALANCED)
        evaluator = ScriptedEvaluator(
            {
                "tp1": saturating(1000, 64, base_latency=2000),
                "tp2": saturating(900, 64, base_latency=500),
                "tp4": saturating(300, 64, base_latency=300),
            }
        )
        outcome = await Tuner(evaluator, wl, SETTINGS).tune(planning(dense_8b, wl))
        assert outcome.winner.plan.tensor_parallel_size == 2


class TestFailures:
    async def test_oom_candidate_is_skipped(self, dense_8b: ModelProfile) -> None:
        wl = workload_from_preset("chat")
        oom = CandidateFailure(type=FailureType.OOM, message="CUDA out of memory", stage="startup")
        evaluator = ScriptedEvaluator(
            {"tp2": saturating(800, 64), "tp4": saturating(500, 64)}, failures={"tp1": oom}
        )
        outcome = await Tuner(evaluator, wl, SETTINGS).tune(planning(dense_8b, wl))
        assert outcome.winner.plan.tensor_parallel_size == 2
        failed = [e for e in outcome.evaluations if e.status == "failed"]
        assert (
            failed and failed[0].failure is not None and failed[0].failure.type == FailureType.OOM
        )
        assert failed[0].plan.viability.value == "launch_failed"
        assert any("failed to launch (oom)" in line for line in outcome.winner.rationale)

    async def test_all_candidates_fail(self, dense_8b: ModelProfile) -> None:
        wl = workload_from_preset("chat")
        crash = CandidateFailure(type=FailureType.ENGINE_CRASH, message="boom")
        evaluator = ScriptedEvaluator({}, failures={"tp1": crash, "tp2": crash, "tp4": crash})
        with pytest.raises(NoViablePlanError) as exc:
            await Tuner(evaluator, wl, SETTINGS).tune(planning(dense_8b, wl))
        assert "Every candidate failed" in exc.value.message

    async def test_relaunch_failure_falls_back_to_stage_a_evidence(
        self, dense_8b: ModelProfile
    ) -> None:
        wl = workload_from_preset("chat")

        class RelaunchFailing(ScriptedEvaluator):
            """Every plan launches once; any relaunch fails (e.g. GPU memory not yet released)."""

            async def open(self, plan: CandidatePlan) -> ScriptedSession:
                if plan.id in self.opens:
                    raise CandidateLaunchFailed(
                        CandidateFailure(type=FailureType.OOM, message="relaunch OOM")
                    )
                return await super().open(plan)

        evaluator = RelaunchFailing(
            {"tp1": saturating(1000, 64), "tp2": saturating(800, 64), "tp4": saturating(500, 64)}
        )
        settings = TuningSettings(
            stage_a_requests=8, sweep_requests=8, final_multiplier=1, memory_tuning=False, top_k=1
        )
        # tp4 is benchmarked last and stays open; tp1 (the best) must be relaunched and fails.
        outcome = await Tuner(evaluator, wl, settings).tune(planning(dense_8b, wl))
        winner = outcome.winner
        assert winner.plan.tensor_parallel_size == 1 and winner.benchmarked
        assert winner.final_result is not None and winner.final_result.spec.label == "stage-a"
        assert winner.plan.max_concurrency == winner.final_result.spec.concurrency
        assert any("selecting on stage A results" in n for n in outcome.notes)
        assert any("Final confirmation launch failed" in n for n in outcome.notes)
        failed = [e for e in outcome.evaluations if e.status == "failed"]
        assert {e.stage for e in failed} == {"concurrency", "final"}

    async def test_error_rate_makes_candidate_ineligible(self, dense_8b: ModelProfile) -> None:
        wl = workload_from_preset("chat")

        def flaky(c: int) -> tuple[float, float, float, float]:
            return 5000.0, 300.0, 50.0, 0.25

        evaluator = ScriptedEvaluator(
            {"tp1": flaky, "tp2": saturating(800, 64), "tp4": saturating(500, 64)}
        )
        outcome = await Tuner(evaluator, wl, SETTINGS).tune(planning(dense_8b, wl))
        assert outcome.winner.plan.tensor_parallel_size == 2


class TestSLOAndSweep:
    async def test_slo_unsatisfied_is_reported(self, dense_8b: ModelProfile) -> None:
        wl = workload_from_preset(
            "chat", latency_constraints=LatencyConstraints(max_p95_latency_ms=100)
        )
        evaluator = ScriptedEvaluator(
            {"tp1": saturating(1000, 64), "tp2": saturating(800, 64), "tp4": saturating(500, 64)}
        )
        outcome = await Tuner(evaluator, wl, SETTINGS).tune(planning(dense_8b, wl))
        assert outcome.winner.slo_satisfied is False
        assert any("No tested configuration satisfied" in n for n in outcome.notes)

    async def test_sweep_stops_at_plateau_and_refines(self, dense_8b: ModelProfile) -> None:
        wl = workload_from_preset("chat")
        evaluator = ScriptedEvaluator({"tp1": saturating(1000, 40)}, failures={})
        settings = TuningSettings(
            stage_a_requests=8,
            sweep_requests=8,
            final_multiplier=1,
            sweep_start=4,
            sweep_max=1024,
            memory_tuning=False,
            top_k=1,
        )
        outcome = await Tuner(evaluator, wl, settings).tune(planning(dense_8b, wl, gpus=1))
        sweep_points = [c for pid, c, label in evaluator.benchmarks if label in ("sweep", "refine")]
        assert sweep_points[:4] == [4, 8, 16, 32]
        assert max(sweep_points) <= 128, "sweep must stop once throughput plateaus"
        assert any(label == "refine" for _, _, label in evaluator.benchmarks)
        chosen = outcome.winner.plan.max_concurrency
        assert chosen is not None and 32 <= chosen <= 64

    async def test_memory_tuning_adopts_improvement(self, dense_8b: ModelProfile) -> None:
        wl = workload_from_preset("chat")
        base = generate_candidates(fh.h100(1), dense_8b, wl, [FakeEngine()])
        plan_id = base.candidates[0].id

        def better(c: int) -> tuple[float, float, float, float]:
            tps, p95, ttft, err = saturating(1000, 40)(c)
            return tps * 1.2, p95, ttft, err

        evaluator = ScriptedEvaluator({"tp1": saturating(1000, 40), f"{plan_id}-mem96": better})
        settings = TuningSettings(
            stage_a_requests=8, sweep_requests=8, final_multiplier=1, memory_tuning=True, top_k=1
        )
        outcome = await Tuner(evaluator, wl, settings).tune(base)
        assert outcome.winner.plan.memory_fraction == pytest.approx(0.96)
        assert any("adopted" in line for line in outcome.winner.rationale)

    async def test_memory_tuning_reverts_on_oom(self, dense_8b: ModelProfile) -> None:
        wl = workload_from_preset("chat")
        base = generate_candidates(fh.h100(1), dense_8b, wl, [FakeEngine()])
        plan_id = base.candidates[0].id
        oom = CandidateFailure(type=FailureType.OOM, message="OOM at higher fraction")
        evaluator = ScriptedEvaluator(
            {"tp1": saturating(1000, 40)}, failures={f"{plan_id}-mem96": oom}
        )
        settings = TuningSettings(
            stage_a_requests=8, sweep_requests=8, final_multiplier=1, memory_tuning=True, top_k=1
        )
        outcome = await Tuner(evaluator, wl, settings).tune(base)
        assert outcome.winner.plan.memory_fraction == base.candidates[0].memory_fraction
        assert any("failed to launch (oom)" in line for line in outcome.winner.rationale)

    async def test_resume_reuses_completed_structural_results(self, dense_8b: ModelProfile) -> None:
        wl = workload_from_preset("chat")
        evaluator = ScriptedEvaluator(
            {"tp1": saturating(1000, 64), "tp2": saturating(800, 64), "tp4": saturating(500, 64)}
        )
        first = await Tuner(evaluator, wl, SETTINGS).tune(planning(dense_8b, wl))
        completed = [e for e in first.evaluations if e.stage == "structural"]
        evaluator2 = ScriptedEvaluator(
            {"tp1": saturating(1000, 64), "tp2": saturating(800, 64), "tp4": saturating(500, 64)}
        )
        await Tuner(evaluator2, wl, SETTINGS, completed=completed).tune(planning(dense_8b, wl))
        assert not any(label == "stage-a" for _, _, label in evaluator2.benchmarks)

        changed = planning(dense_8b, wl)
        changed.candidates[0].memory_fraction = 0.75
        changed.candidates[0].engine_args = {"seed": 7}
        evaluator3 = ScriptedEvaluator(
            {"tp1": saturating(1000, 64), "tp2": saturating(800, 64), "tp4": saturating(500, 64)}
        )
        await Tuner(evaluator3, wl, SETTINGS, completed=completed).tune(changed)
        rerun = [cid for cid, _, label in evaluator3.benchmarks if label == "stage-a"]
        assert rerun == [changed.candidates[0].id]

    async def test_progress_and_persistence_callbacks(self, dense_8b: ModelProfile) -> None:
        wl = workload_from_preset("chat")
        evaluator = ScriptedEvaluator({"tp1": saturating(1000, 64)})
        persisted: list[str] = []
        events: list[str] = []

        class Progress:
            def stage(self, name: str, detail: str) -> None:
                events.append(f"stage:{name}")

            def candidate_started(self, plan: CandidatePlan, detail: str) -> None:
                events.append("started")

            def candidate_result(self, plan: CandidatePlan, result: BenchmarkResult) -> None:
                events.append("result")

            def candidate_failed(self, plan: CandidatePlan, failure: CandidateFailure) -> None:
                events.append("failed")

            def note(self, message: str) -> None:
                events.append("note")

        await Tuner(
            evaluator,
            wl,
            SETTINGS,
            progress=Progress(),
            on_evaluation=lambda e: persisted.append(e.stage),
        ).tune(planning(dense_8b, wl, gpus=1))
        assert "stage:A" in events and "stage:B" in events and "stage:D" in events
        assert persisted[0] == "structural" and persisted[-1] == "final"
