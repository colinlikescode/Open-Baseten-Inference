"""The plan → tune → serve pipeline shared by the CLI commands."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from servepilot import __version__
from servepilot.benchmark.evaluator import LaunchingEvaluator
from servepilot.benchmark.tuner import Tuner, TuningOutcome, TuningSettings
from servepilot.cache.fingerprints import (
    CacheValidation,
    hardware_fingerprint,
    model_fingerprint,
    validate_record,
    workload_fingerprint,
)
from servepilot.cli.common import Workspace, hf_token
from servepilot.cli.render import summarize_result
from servepilot.constants import DEFAULT_STARTUP_STAGGER_SECONDS
from servepilot.exceptions import CacheError, NoViablePlanError
from servepilot.logging import get_logger
from servepilot.models.tokenizer import load_tokenizer
from servepilot.planner.explain import explain_plan
from servepilot.planner.memory import MemoryModelConfig
from servepilot.planner.planner import Planner
from servepilot.runtime.replicas import ReplicaSet
from servepilot.runtime.router import ReplicaRouter
from servepilot.schemas.benchmark import BenchmarkResult, CandidateFailure
from servepilot.schemas.plan import CandidateEvaluation, CandidatePlan, PlanningResult, SelectedPlan
from servepilot.schemas.runtime import TuningRecord, make_record_key

log = get_logger(__name__)


# --------------------------------------------------------------------------- planning
def run_plan(ws: Workspace) -> tuple[PlanningResult, list[str]]:
    planner = Planner(
        engines=ws.engines, constraints=ws.constraints, memory_cfg=MemoryModelConfig()
    )
    result = planner.plan(ws.hardware, ws.model, ws.workload)
    explanation = explain_plan(result, ws.model, ws.hardware)
    return result, explanation


# --------------------------------------------------------------------------- cache
@dataclass
class CacheLookup:
    record: TuningRecord | None
    validation: CacheValidation | None
    key: str


def fingerprints(ws: Workspace, planning: PlanningResult) -> tuple[str, str, str]:
    return (
        hardware_fingerprint(ws.hardware, planning.selected_gpu_ids),
        model_fingerprint(ws.model),
        workload_fingerprint(ws.workload),
    )


def lookup_cache(ws: Workspace, planning: PlanningResult) -> CacheLookup:
    hw_fp, model_fp, wl_fp = fingerprints(ws, planning)
    key = make_record_key(hw_fp, model_fp, wl_fp)
    try:
        record = ws.cache.find(hw_fp, model_fp, wl_fp)
    except CacheError as exc:
        log.warning("ignoring unreadable cache record: %s", exc)
        return CacheLookup(None, None, key)
    if record is None:
        return CacheLookup(None, None, key)
    validation = validate_record(
        record,
        hardware=ws.hardware,
        model=ws.model,
        workload=ws.workload,
        engine_versions=ws.registry.versions(),
        gpu_ids=planning.selected_gpu_ids,
        planning=planning,
        constraints=ws.constraints,
    )
    return CacheLookup(record, validation, key)


# --------------------------------------------------------------------------- tuning
class RichTuningProgress:
    def __init__(self, console: Console) -> None:
        self._console = console
        self._t0 = time.monotonic()

    def _stamp(self) -> str:
        return f"[dim]{time.monotonic() - self._t0:6.0f}s[/]"

    def stage(self, name: str, detail: str) -> None:
        self._console.print(f"\n{self._stamp()} [bold cyan]Stage {name}[/] {detail}")

    def candidate_started(self, plan: CandidatePlan, detail: str) -> None:
        self._console.print(f"{self._stamp()}   {plan.label()}: {detail}")

    def candidate_result(self, plan: CandidatePlan, result: BenchmarkResult) -> None:
        self._console.print(
            f"{self._stamp()}   [green]✓[/] {plan.label()} @c={result.spec.concurrency}: {summarize_result(result)}"
        )

    def candidate_failed(self, plan: CandidatePlan, failure: CandidateFailure) -> None:
        first = failure.message.splitlines()[0] if failure.message else failure.type.value
        self._console.print(
            f"{self._stamp()}   [red]✗[/] {plan.label()} failed ({failure.type.value}): {first[:160]}"
        )

    def note(self, message: str) -> None:
        self._console.print(f"{self._stamp()}   [dim]{message}[/]")


@dataclass
class TuneRun:
    outcome: TuningOutcome
    record: TuningRecord
    path: Any = None
    notes: list[str] = field(default_factory=list)


def tuning_settings(ws: Workspace) -> TuningSettings:
    t = ws.config.tuning
    return TuningSettings(
        max_structural_candidates=t.max_structural_candidates,
        top_k=t.top_k,
        stage_a_requests=t.stage_a_requests,
        sweep_requests=t.sweep_requests,
        final_multiplier=t.final_multiplier,
        memory_tuning=t.memory_tuning and ws.constraints.memory_fraction is None,
        seed=t.seed,
    )


async def run_tune(
    ws: Workspace,
    planning: PlanningResult,
    *,
    console: Console,
    resume: bool = False,
    verify_gpu_cleanup: bool = True,
) -> TuneRun:
    hw_fp, model_fp, wl_fp = fingerprints(ws, planning)
    settings = tuning_settings(ws)
    versions = ws.registry.versions()
    completed: list[CandidateEvaluation] = []
    record: TuningRecord | None = None
    if resume:
        try:
            existing = await asyncio.to_thread(ws.cache.find, hw_fp, model_fp, wl_fp)
        except CacheError:
            existing = None
        if (
            existing is not None
            and existing.status == "in_progress"
            and existing.seed == settings.seed
        ):
            completed = [
                e
                for e in existing.candidates
                if e.stage == "structural"
                and e.status == "benchmarked"
                and e.engine_version is not None
                and e.engine_version == versions.get(e.plan.engine.value)
            ]
            record = existing
            record.engine_versions = versions
            console.print(
                f"Resuming interrupted tuning run with {len(completed)} completed structural candidate(s)."
            )
    if record is None:
        record = TuningRecord(
            servepilot_version=__version__,
            hardware_fingerprint=hw_fp,
            model_fingerprint=model_fp,
            workload_fingerprint=wl_fp,
            engine_versions=versions,
            hardware_snapshot=ws.hardware,
            model_profile=ws.model,
            workload_profile=ws.workload,
            seed=settings.seed,
            status="in_progress",
            tuning_settings={
                "max_structural_candidates": settings.max_structural_candidates,
                "top_k": settings.top_k,
                "stage_a_requests": settings.stage_a_requests,
                "sweep_requests": settings.sweep_requests,
                "final_multiplier": settings.final_multiplier,
                "memory_tuning": settings.memory_tuning,
                "plateau_min_gain": settings.scoring.plateau_min_gain,
                "plateau_latency_rise": settings.scoring.plateau_latency_rise,
                "max_error_rate": settings.scoring.max_error_rate,
            },
        )
    record.candidates = list(completed)

    # Tokenizer loading may download from the Hub and parse a large tokenizer.json.
    tokenizer = await asyncio.to_thread(
        load_tokenizer,
        ws.model,
        token=hf_token(),
        trust_remote_code=ws.constraints.trust_remote_code,
    )
    if not tokenizer.exact:
        record.notes.append("token counts are approximate: no tokenizer could be loaded")
    evaluator = LaunchingEvaluator(
        registry=ws.registry,
        launcher=ws.launcher,
        ports=ws.ports,
        hardware_provider=ws.hardware_provider,
        hardware=ws.hardware,
        model=ws.model,
        workload=ws.workload,
        tokenizer=tokenizer,
        served_model_name=ws.served_model_name,
        trust_remote_code=ws.constraints.trust_remote_code,
        startup_timeout=ws.config.tuning.startup_timeout_seconds,
        stagger_seconds=DEFAULT_STARTUP_STAGGER_SECONDS,
        verify_gpu_cleanup=verify_gpu_cleanup,
    )

    async def save() -> Path:
        # Serialising the record and fsyncing it must not stall the router or engine log pumps.
        return await asyncio.to_thread(ws.cache.save, record)

    async def persist(evaluation: CandidateEvaluation) -> None:
        if evaluation not in record.candidates:
            record.candidates.append(evaluation)
        await save()

    tuner = Tuner(
        evaluator,
        ws.workload,
        settings,
        progress=RichTuningProgress(console),
        on_evaluation=persist,
        completed=completed,
    )
    try:
        outcome = await tuner.tune(planning)
    except NoViablePlanError:
        # Definitive for this hardware/model/workload: nothing to resume.
        record.status = "failed"
        await save()
        raise
    except BaseException:
        # Interrupted or crashed part-way: keep finished candidates resumable.
        record.status = "in_progress" if tuner.evaluations else "failed"
        await save()
        raise
    finally:
        await ws.launcher.shutdown_all()
    outcome.winner.equivalent_commands = equivalent_commands(ws, outcome.winner)
    record.winner = outcome.winner
    record.candidates = list(outcome.evaluations)
    record.notes.extend(outcome.notes)
    record.status = "complete"
    path = await save()
    return TuneRun(outcome=outcome, record=record, path=path, notes=outcome.notes)


# --------------------------------------------------------------------------- helpers
def equivalent_commands(ws: Workspace, selected: SelectedPlan) -> list[str]:
    """Render the backend launch commands for the selected plan (secrets redacted)."""
    engine = ws.registry.get(selected.plan.engine)
    if not engine.is_available():
        return []
    replica_set = ReplicaSet(
        plan=selected.plan,
        model=ws.model,
        engine=engine,
        launcher=ws.launcher,
        ports=ws.ports,
        hardware=ws.hardware,
        host=ws.config.server.host,
        served_model_name=ws.served_model_name,
        trust_remote_code=ws.constraints.trust_remote_code,
        router=ReplicaRouter(),
    )
    return [spec.redacted_display_command for spec in replica_set.specs()]
