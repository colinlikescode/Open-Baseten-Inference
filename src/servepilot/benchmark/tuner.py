"""Staged empirical tuner.

Stage A – structural topology search: launch each viable candidate (engine × TP × replicas ×
          MoE variant) and benchmark it under a saturating (or the expected) load.
Stage B – concurrency sweep on the best few candidates: geometric growth with plateau/SLO/error
          stopping and local refinement around the best region.
Stage C – memory tuning: try a larger engine memory fraction for the winner; keep it only when it
          launches and measurably helps.
Stage D – final confirmation: re-benchmark the winner with a larger sample; persist those metrics.

Failures (OOM, crashes, timeouts) are recorded per candidate and never abort the search.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from servepilot.benchmark.evaluator import (
    CandidateEvaluator,
    CandidateLaunchFailed,
    CandidateSession,
)
from servepilot.benchmark.runner import make_spec
from servepilot.constants import (
    DEFAULT_BENCHMARK_SEED,
    DEFAULT_FINAL_CONFIRMATION_MULTIPLIER,
    DEFAULT_MAX_STRUCTURAL_CANDIDATES,
    DEFAULT_STAGE_A_REQUESTS,
    DEFAULT_SWEEP_GROWTH_FACTOR,
    DEFAULT_SWEEP_MAX_CONCURRENCY,
    DEFAULT_SWEEP_REFINEMENT_POINTS,
    DEFAULT_SWEEP_REQUESTS_PER_POINT,
    DEFAULT_SWEEP_START_CONCURRENCY,
    DEFAULT_TOP_K_FOR_CONCURRENCY_SWEEP,
    MEMORY_TUNING_MIN_GAIN,
    MEMORY_TUNING_STEP,
    MIN_HEADROOM_BYTES,
)
from servepilot.exceptions import BenchmarkError, NoViablePlanError
from servepilot.logging import get_logger
from servepilot.planner.explain import build_rationale
from servepilot.planner.scoring import (
    ScoringConfig,
    choose_concurrency,
    pareto_front,
    score_results,
    select_best,
    should_continue_sweep,
)
from servepilot.schemas.benchmark import BenchmarkResult, BenchmarkSpec, CandidateFailure
from servepilot.schemas.plan import (
    CandidateEvaluation,
    CandidatePlan,
    CandidateViability,
    PlanningResult,
    SelectedPlan,
)
from servepilot.schemas.workload import (
    DEFAULT_LATENCY_CONCURRENCY,
    LatencyConstraints,
    Objective,
    WorkloadProfile,
)

log = get_logger(__name__)

STAGE_A_MIN_CONCURRENCY = 8
STAGE_A_MAX_CONCURRENCY = 256


@dataclass(frozen=True)
class TuningSettings:
    max_structural_candidates: int = DEFAULT_MAX_STRUCTURAL_CANDIDATES
    top_k: int = DEFAULT_TOP_K_FOR_CONCURRENCY_SWEEP
    stage_a_requests: int = DEFAULT_STAGE_A_REQUESTS
    sweep_requests: int = DEFAULT_SWEEP_REQUESTS_PER_POINT
    final_multiplier: int = DEFAULT_FINAL_CONFIRMATION_MULTIPLIER
    memory_tuning: bool = True
    seed: int = DEFAULT_BENCHMARK_SEED
    sweep_start: int = DEFAULT_SWEEP_START_CONCURRENCY
    sweep_max: int = DEFAULT_SWEEP_MAX_CONCURRENCY
    sweep_growth: int = DEFAULT_SWEEP_GROWTH_FACTOR
    refinement_points: int = DEFAULT_SWEEP_REFINEMENT_POINTS
    memory_step: float = MEMORY_TUNING_STEP
    memory_min_gain: float = MEMORY_TUNING_MIN_GAIN
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    request_timeout_seconds: float = 300.0


class TuningProgress(Protocol):
    def stage(self, name: str, detail: str) -> None: ...

    def candidate_started(self, plan: CandidatePlan, detail: str) -> None: ...

    def candidate_result(self, plan: CandidatePlan, result: BenchmarkResult) -> None: ...

    def candidate_failed(self, plan: CandidatePlan, failure: CandidateFailure) -> None: ...

    def note(self, message: str) -> None: ...


class NullProgress:
    def stage(self, name: str, detail: str) -> None:
        log.info("stage %s: %s", name, detail)

    def candidate_started(self, plan: CandidatePlan, detail: str) -> None:
        log.info("%s: %s", plan.label(), detail)

    def candidate_result(self, plan: CandidatePlan, result: BenchmarkResult) -> None:
        log.info("%s: %s", plan.label(), result.short())

    def candidate_failed(self, plan: CandidatePlan, failure: CandidateFailure) -> None:
        log.warning("%s failed (%s): %s", plan.label(), failure.type.value, failure.message)

    def note(self, message: str) -> None:
        log.info(message)


@dataclass
class TuningOutcome:
    winner: SelectedPlan
    evaluations: list[CandidateEvaluation]
    notes: list[str] = field(default_factory=list)


# Called after every recorded evaluation (persistence hook). May be a plain function or return an
# awaitable, so callers can do their disk I/O off the event loop.
EvaluationCallback = Callable[[CandidateEvaluation], Awaitable[None] | None]


class Tuner:
    def __init__(
        self,
        evaluator: CandidateEvaluator,
        workload: WorkloadProfile,
        settings: TuningSettings | None = None,
        *,
        progress: TuningProgress | None = None,
        on_evaluation: EvaluationCallback | None = None,
        completed: Sequence[CandidateEvaluation] = (),
    ) -> None:
        self._eval = evaluator
        self._workload = workload
        self._s = settings or TuningSettings()
        self._progress: TuningProgress = progress or NullProgress()
        self._on_evaluation = on_evaluation
        self._completed = {
            e.plan.id: e for e in completed if e.stage == "structural" and e.status == "benchmarked"
        }
        self.evaluations: list[CandidateEvaluation] = []
        self.notes: list[str] = []
        self._session: CandidateSession | None = None

    # ------------------------------------------------------------------ helpers
    @property
    def objective(self) -> Objective:
        return self._workload.objective

    @property
    def slo(self) -> LatencyConstraints | None:
        return self._workload.latency_constraints

    async def _record(self, evaluation: CandidateEvaluation) -> None:
        self.evaluations.append(evaluation)
        if self._on_evaluation is not None:
            outcome = self._on_evaluation(evaluation)
            if inspect.isawaitable(outcome):
                await outcome

    def _spec(self, concurrency: int, requests: int, label: str) -> BenchmarkSpec:
        return make_spec(
            self._workload,
            concurrency=concurrency,
            num_requests=max(requests, 2 * concurrency),
            seed=self._s.seed,
            label=label,
            request_rate=self._workload.target_request_rate
            if self.objective == Objective.LATENCY
            else None,
            timeout_seconds=self._s.request_timeout_seconds,
        )

    def _stage_a_concurrency(self, plan: CandidatePlan) -> int:
        if self._workload.expected_concurrency is not None:
            return self._workload.expected_concurrency
        if self.objective == Objective.LATENCY:
            self._note_once(
                f"No expected concurrency given for the latency objective; benchmarking at {DEFAULT_LATENCY_CONCURRENCY} concurrent requests."
            )
            return DEFAULT_LATENCY_CONCURRENCY
        cap = plan.max_concurrency or (plan.max_num_seqs or 64) * plan.replica_count
        # Half the estimated capacity keeps every replica busy without overloading the engine;
        # each replica gets at least two concurrent requests so multi-replica plans are not starved.
        return max(
            STAGE_A_MIN_CONCURRENCY, 2 * plan.replica_count, min(STAGE_A_MAX_CONCURRENCY, cap // 2)
        )

    def _note_once(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)
            self._progress.note(message)

    async def _open(
        self, plan: CandidatePlan
    ) -> tuple[CandidateSession | None, CandidateFailure | None]:
        """Open a session, reusing the currently open one when it serves the same plan."""
        if self._session is not None:
            if (
                self._session.plan.structural_key() == plan.structural_key()
                and self._session.plan.memory_fraction == plan.memory_fraction
            ):
                return self._session, None
            await self._close_session()
        try:
            self._session = await self._eval.open(plan)
        except CandidateLaunchFailed as exc:
            self._progress.candidate_failed(plan, exc.failure)
            return None, exc.failure
        return self._session, None

    async def _close_session(self) -> None:
        if self._session is None:
            return
        session, self._session = self._session, None
        await session.close()

    def _score(self, results: Sequence[BenchmarkResult]) -> None:
        for s in score_results(results, self.objective, self.slo, self._s.scoring):
            s.result.score = s.score

    # ------------------------------------------------------------------ main entry
    async def tune(self, planning: PlanningResult) -> TuningOutcome:
        candidates = list(planning.viable)[: self._s.max_structural_candidates]
        if not candidates:
            raise NoViablePlanError("no viable candidates to tune")
        try:
            structural = await self._stage_a(candidates)
            if not structural:
                failures = [e for e in self.evaluations if e.status == "failed"]
                summary = "\n".join(
                    f"  - {e.plan.label()}: {e.failure.message if e.failure else 'unknown'}"
                    for e in failures
                )
                raise NoViablePlanError(
                    "Every candidate failed to launch or benchmark.\n" + summary,
                    hints=[
                        "Run with -vv to see engine logs.",
                        "Try a smaller context (--context-length) or a lower --memory-headroom.",
                    ],
                )
            sweeps = await self._stage_b(structural)
            best_eval, best_result, reason = self._pick_winner(sweeps)
            best_eval, best_result, memory_note = await self._stage_c(best_eval, best_result)
            final_eval, final_result = await self._stage_d(best_eval, best_result)
        finally:
            await self._close_session()

        winner_plan = final_eval.plan.with_updates(
            max_concurrency=final_result.spec.concurrency,
            viability=CandidateViability.LAUNCH_VALIDATED,
        )
        all_results = [r for e in self.evaluations for r in e.results]
        scored = score_results([final_result], self.objective, self.slo, self._s.scoring)
        slo_ok = scored[0].slo_satisfied if self._workload.has_slo else None
        sweep_hist = [
            r
            for e in self.evaluations
            if e.plan.structural_key() == winner_plan.structural_key()
            for r in e.results
        ]
        smaller_excluded = any(
            (ex.tensor_parallel_size or 0) < winner_plan.tensor_parallel_size
            and "memory" in ex.reason.lower()
            for ex in planning.excluded
        ) and not any(
            e.plan.tensor_parallel_size < winner_plan.tensor_parallel_size
            and e.status == "benchmarked"
            for e in self.evaluations
        )
        rationale = build_rationale(
            winner=winner_plan,
            evaluations=self.evaluations,
            objective=self.objective,
            sweep_history=sweep_hist,
            chosen_concurrency_reason=reason,
            final_result=final_result,
            slo_satisfied=slo_ok,
            memory_tuning_note=memory_note,
            smaller_tp_excluded=smaller_excluded,
        )
        selected = SelectedPlan(
            plan=winner_plan,
            objective=self.objective,
            benchmarked=True,
            slo_satisfied=slo_ok,
            final_result=final_result,
            rationale=rationale,
            engine_version=final_eval.engine_version,
            pareto_front=pareto_front(all_results),
            source="tuned",
        )
        return TuningOutcome(winner=selected, evaluations=self.evaluations, notes=list(self.notes))

    # ------------------------------------------------------------------ stage A
    async def _stage_a(self, candidates: list[CandidatePlan]) -> list[CandidateEvaluation]:
        self._progress.stage("A", f"structural topology search over {len(candidates)} candidate(s)")
        benchmarked: list[CandidateEvaluation] = []
        for plan in candidates:
            if plan.id in self._completed:
                prior = self._completed[plan.id]
                self._progress.note(
                    f"reusing completed result for {plan.label()} from interrupted run"
                )
                self.evaluations.append(prior)
                benchmarked.append(prior)
                continue
            evaluation = CandidateEvaluation(plan=plan, stage="structural")
            concurrency = self._stage_a_concurrency(plan)
            self._progress.candidate_started(
                plan, f"launching, then benchmarking at concurrency {concurrency}"
            )
            session, failure = await self._open(plan)
            if session is None:
                evaluation.status = "failed"
                evaluation.failure = failure
                plan.viability = CandidateViability.LAUNCH_FAILED
                await self._record(evaluation)
                continue
            evaluation.launch_seconds = session.launch_seconds
            evaluation.engine_version = session.engine_version
            evaluation.runtime_metadata = dict(session.runtime_metadata)
            plan.viability = CandidateViability.LAUNCH_VALIDATED
            try:
                result = await session.benchmark(
                    self._spec(concurrency, self._s.stage_a_requests, "stage-a")
                )
            except BenchmarkError as exc:
                evaluation.status = "failed"
                evaluation.failure = CandidateFailure(message=str(exc), stage="benchmark")
                self._progress.candidate_failed(plan, evaluation.failure)
                await self._record(evaluation)
                await self._close_session()
                continue
            evaluation.results.append(result)
            evaluation.status = "benchmarked"
            self._progress.candidate_result(plan, result)
            await self._record(evaluation)
            benchmarked.append(evaluation)
            # Keep the last session open: stage B often starts with this candidate.
            if plan is not candidates[-1]:
                await self._close_session()
        self._score([e.results[0] for e in benchmarked])
        return benchmarked

    # ------------------------------------------------------------------ stage B
    def _rank_structural(self, evaluations: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
        scored = score_results(
            [e.results[0] for e in evaluations], self.objective, self.slo, self._s.scoring
        )
        order = sorted(
            zip(evaluations, scored, strict=True),
            key=lambda pair: (not pair[1].eligible, -pair[1].score),
        )
        return [e for e, _ in order]

    def _sweep_points(self, plan: CandidatePlan) -> list[int]:
        upper = min(self._s.sweep_max, plan.max_concurrency or self._s.sweep_max)
        points: list[int] = []
        c = self._s.sweep_start
        while c <= upper:
            points.append(c)
            c *= self._s.sweep_growth
        if not points or points[-1] < upper:
            points.append(upper)
        return sorted(set(points))

    async def _stage_b(self, structural: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
        ranked = self._rank_structural(structural)[: self._s.top_k]
        # Reorder so the candidate whose session is still open goes first (avoids a relaunch).
        open_session = self._session
        if open_session is not None:
            open_key = open_session.plan.structural_key()
            ranked.sort(key=lambda e: e.plan.structural_key() != open_key)
        self._progress.stage("B", f"concurrency tuning on top {len(ranked)} candidate(s)")
        sweeps: list[CandidateEvaluation] = []
        for structural_eval in ranked:
            plan = structural_eval.plan
            evaluation = CandidateEvaluation(
                plan=plan, stage="concurrency", engine_version=structural_eval.engine_version
            )
            session, failure = await self._open(plan)
            if session is None:
                evaluation.status = "failed"
                evaluation.failure = failure
                await self._record(evaluation)
                continue
            history: list[BenchmarkResult] = list(structural_eval.results)
            if self.objective == Objective.LATENCY:
                await self._sweep_latency(session, plan, history, evaluation)
            else:
                await self._sweep_throughput(session, plan, history, evaluation)
            evaluation.status = "benchmarked" if evaluation.results or history else "failed"
            self._score(history)
            await self._record(evaluation)
            sweeps.append(evaluation)
        if not sweeps:
            # Every top candidate failed to relaunch. Their stage A measurements are still valid
            # evidence, so select on those instead of aborting; stage D relaunches the winner.
            self._note_once(
                "No top candidate could be relaunched for the concurrency sweep; selecting on stage A results."
            )
            return ranked
        return sweeps

    async def _run_point(
        self,
        session: CandidateSession,
        plan: CandidatePlan,
        concurrency: int,
        history: list[BenchmarkResult],
        evaluation: CandidateEvaluation,
        label: str,
    ) -> BenchmarkResult | None:
        if any(r.spec.concurrency == concurrency and r.spec.label != "stage-a" for r in history):
            return None
        self._progress.candidate_started(plan, f"benchmarking at concurrency {concurrency}")
        try:
            result = await session.benchmark(self._spec(concurrency, self._s.sweep_requests, label))
        except BenchmarkError as exc:
            evaluation.notes.append(f"concurrency {concurrency}: {exc}")
            self._progress.note(f"{plan.label()} at concurrency {concurrency}: {exc}")
            return None
        history.append(result)
        evaluation.results.append(result)
        self._progress.candidate_result(plan, result)
        return result

    async def _sweep_throughput(
        self,
        session: CandidateSession,
        plan: CandidatePlan,
        history: list[BenchmarkResult],
        evaluation: CandidateEvaluation,
    ) -> None:
        points = self._sweep_points(plan)
        sweep_history: list[BenchmarkResult] = []
        for c in points:
            result = await self._run_point(session, plan, c, history, evaluation, "sweep")
            if result is None:
                continue
            sweep_history.append(result)
            decision = should_continue_sweep(sweep_history, self.slo, self._s.scoring)
            if not decision.continue_sweep:
                evaluation.notes.append(f"sweep stopped at {c}: {decision.reason}")
                self._progress.note(f"{plan.label()}: {decision.reason}")
                break
        # Local refinement around the best point.
        valid = sorted(
            (r for r in history if r.successful_requests > 0), key=lambda r: r.spec.concurrency
        )
        if len(valid) >= 2 and self._s.refinement_points > 0:
            best = max(valid, key=lambda r: r.output_tokens_per_second)
            tested = {r.spec.concurrency for r in valid}
            idx = valid.index(best)
            neighbours: list[int] = []
            if idx + 1 < len(valid):
                neighbours.append((best.spec.concurrency + valid[idx + 1].spec.concurrency) // 2)
            if idx > 0:
                neighbours.append((best.spec.concurrency + valid[idx - 1].spec.concurrency) // 2)
            for c in neighbours[: self._s.refinement_points]:
                if c in tested or c <= 0:
                    continue
                await self._run_point(session, plan, c, history, evaluation, "refine")

    async def _sweep_latency(
        self,
        session: CandidateSession,
        plan: CandidatePlan,
        history: list[BenchmarkResult],
        evaluation: CandidateEvaluation,
    ) -> None:
        base = self._workload.expected_concurrency or DEFAULT_LATENCY_CONCURRENCY
        # More samples at the operating concurrency.
        await self._run_point(session, plan, base, history, evaluation, "latency")
        if not self._workload.has_slo:
            return
        # Under an SLO, find the highest concurrency that still meets it.
        c = base * self._s.sweep_growth
        upper = min(self._s.sweep_max, plan.max_concurrency or self._s.sweep_max)
        sweep_history = [r for r in history if r.spec.concurrency == base]
        while c <= upper:
            result = await self._run_point(session, plan, c, history, evaluation, "sweep")
            if result is None:
                break
            sweep_history.append(result)
            decision = should_continue_sweep(sweep_history, self.slo, self._s.scoring)
            if not decision.continue_sweep:
                evaluation.notes.append(f"sweep stopped at {c}: {decision.reason}")
                break
            c *= self._s.sweep_growth

    # ------------------------------------------------------------------ selection
    def _pick_winner(
        self, sweeps: list[CandidateEvaluation]
    ) -> tuple[CandidateEvaluation, BenchmarkResult, str]:
        operating: list[tuple[CandidateEvaluation, BenchmarkResult, str]] = []
        for evaluation in sweeps:
            history = [
                r
                for e in self.evaluations
                if e.plan.structural_key() == evaluation.plan.structural_key()
                for r in e.results
            ]
            choice = choose_concurrency(history, self.objective, self.slo, self._s.scoring)
            if choice is None:
                continue
            operating.append((evaluation, choice.result, choice.reason))
        if not operating:
            raise NoViablePlanError("no candidate produced a valid benchmark result")
        scored = score_results(
            [r for _, r, _ in operating], self.objective, self.slo, self._s.scoring
        )
        selection = select_best(scored)
        if selection is None:
            raise NoViablePlanError(
                "every benchmarked candidate exceeded the acceptable error rate"
            )
        for note in selection.notes:
            self._note_once(note)
        for (evaluation, result, reason), s in zip(operating, scored, strict=True):
            result.score = s.score
            if s is selection.winner:
                return evaluation, result, reason
        raise NoViablePlanError("winner selection failed")  # pragma: no cover - defensive

    # ------------------------------------------------------------------ stage C
    async def _stage_c(
        self, best: CandidateEvaluation, best_result: BenchmarkResult
    ) -> tuple[CandidateEvaluation, BenchmarkResult, str | None]:
        if not self._s.memory_tuning or self.objective == Objective.LATENCY:
            return best, best_result, None
        plan = best.plan
        est = plan.estimated_memory
        if plan.memory_fraction is None or est is None:
            return best, best_result, None
        ceiling = (
            math.floor((est.device_free_bytes - MIN_HEADROOM_BYTES) / est.device_total_bytes * 100)
            / 100
        )
        new_fraction = round(min(plan.memory_fraction + self._s.memory_step, ceiling), 2)
        if new_fraction <= plan.memory_fraction:
            return best, best_result, None
        self._progress.stage(
            "C",
            f"memory tuning: trying memory fraction {new_fraction:.2f} (was {plan.memory_fraction:.2f})",
        )
        tuned_plan = plan.with_updates(
            id=f"{plan.id}-mem{int(new_fraction * 100)}", memory_fraction=new_fraction
        )
        tuned_plan.rationale = list(plan.rationale)
        evaluation = CandidateEvaluation(
            plan=tuned_plan, stage="memory", engine_version=best.engine_version
        )
        await self._close_session()
        session, failure = await self._open(tuned_plan)
        if session is None:
            evaluation.status = "failed"
            evaluation.failure = failure
            await self._record(evaluation)
            note = f"Memory fraction {new_fraction:.2f} failed to launch ({failure.type.value if failure else 'unknown'}); keeping {plan.memory_fraction:.2f}."
            self._progress.note(note)
            return best, best_result, note
        history: list[BenchmarkResult] = []
        c0 = best_result.spec.concurrency
        for c in (c0, c0 * self._s.sweep_growth):
            if plan.max_concurrency and c > plan.max_concurrency * 2:
                break
            await self._run_point(session, tuned_plan, c, history, evaluation, "memory")
        evaluation.status = "benchmarked" if history else "failed"
        self._score(history)
        await self._record(evaluation)
        choice = choose_concurrency(history, self.objective, self.slo, self._s.scoring)
        if choice is None:
            return (
                best,
                best_result,
                f"Memory fraction {new_fraction:.2f} produced no valid result; keeping {plan.memory_fraction:.2f}.",
            )
        pair = score_results(
            [best_result, choice.result], self.objective, self.slo, self._s.scoring
        )
        gain = (pair[1].score - pair[0].score) / max(pair[0].score, 1e-9)
        if pair[1].eligible and gain >= self._s.memory_min_gain:
            note = f"Raising the memory fraction to {new_fraction:.2f} improved the {self.objective.value} score by {gain:.1%}; adopted."
            self._progress.note(note)
            choice.result.score = pair[1].score
            return evaluation, choice.result, note
        note = f"Raising the memory fraction to {new_fraction:.2f} changed the score by {gain:+.1%} (< {self._s.memory_min_gain:.0%}); keeping {plan.memory_fraction:.2f}."
        self._progress.note(note)
        await self._close_session()
        return best, best_result, note

    # ------------------------------------------------------------------ stage D
    async def _stage_d(
        self, best: CandidateEvaluation, best_result: BenchmarkResult
    ) -> tuple[CandidateEvaluation, BenchmarkResult]:
        plan = best.plan.with_updates(max_concurrency=best_result.spec.concurrency)
        requests = self._s.sweep_requests * self._s.final_multiplier
        self._progress.stage(
            "D",
            f"final confirmation of {plan.label()} at concurrency {plan.max_concurrency} with {max(requests, 2 * (plan.max_concurrency or 1))} requests",
        )
        evaluation = CandidateEvaluation(
            plan=plan, stage="final", engine_version=best.engine_version
        )
        session, failure = await self._open(plan)
        if session is None:
            evaluation.status = "failed"
            evaluation.failure = failure
            await self._record(evaluation)
            self._note_once(
                "Final confirmation launch failed; using the preliminary measurement as the final result."
            )
            return best, best_result
        evaluation.launch_seconds = session.launch_seconds
        evaluation.runtime_metadata = dict(session.runtime_metadata)
        try:
            result = await session.benchmark(
                self._spec(plan.max_concurrency or 1, requests, "final")
            )
        except BenchmarkError as exc:
            evaluation.status = "failed"
            evaluation.failure = CandidateFailure(message=str(exc), stage="final")
            await self._record(evaluation)
            self._note_once(
                f"Final confirmation benchmark failed ({exc}); using the preliminary measurement."
            )
            return best, best_result
        self._score([result])
        evaluation.results.append(result)
        evaluation.status = "benchmarked"
        self._progress.candidate_result(plan, result)
        await self._record(evaluation)
        return evaluation, result
