"""Planner facade: hardware + model + workload → candidate plans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from servepilot.engines.base import InferenceEngine
from servepilot.planner.candidates import generate_candidates
from servepilot.planner.explain import explain_plan
from servepilot.planner.memory import MemoryModelConfig
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidatePlan, PlanConstraints, PlanningResult, SelectedPlan
from servepilot.schemas.workload import WorkloadProfile


@dataclass
class Planner:
    engines: Sequence[InferenceEngine]
    constraints: PlanConstraints = field(default_factory=PlanConstraints)
    memory_cfg: MemoryModelConfig = field(default_factory=MemoryModelConfig)

    def plan(
        self, hardware: HardwareSnapshot, model: ModelProfile, workload: WorkloadProfile
    ) -> PlanningResult:
        return generate_candidates(
            hardware, model, workload, self.engines, self.constraints, self.memory_cfg
        )

    def explain(
        self, result: PlanningResult, model: ModelProfile, hardware: HardwareSnapshot
    ) -> list[str]:
        return explain_plan(result, model, hardware)

    @staticmethod
    def heuristic_selection(result: PlanningResult, workload: WorkloadProfile) -> SelectedPlan:
        """Best-ranked viable candidate, explicitly labelled as unbenchmarked (``--no-tune``)."""
        viable = result.viable
        if not viable:
            raise ValueError("no viable candidates to select from")
        plan: CandidatePlan = min(
            viable, key=lambda p: p.heuristic_rank if p.heuristic_rank is not None else 1_000
        )
        return SelectedPlan(
            plan=plan,
            objective=workload.objective,
            benchmarked=False,
            rationale=[
                "UNBENCHMARKED heuristic plan (--no-tune): chosen by static estimates only.",
                *plan.rationale,
            ],
            source="heuristic",
        )
