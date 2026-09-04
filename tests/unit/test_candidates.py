"""Candidate generation, pruning, overrides, heterogeneous hardware and cluster topologies."""

from __future__ import annotations

import pytest

from servepilot.constants import GIB
from servepilot.exceptions import ConfigurationError, HardwareError, NoViablePlanError
from servepilot.planner.candidates import generate_candidates
from servepilot.planner.planner import Planner
from servepilot.planner.pruning import check_tp_divisibility, dedupe, divisors
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidateViability, EngineName, PlanConstraints
from servepilot.schemas.workload import Objective, WorkloadProfile, workload_from_preset
from servepilot.testing import fake_hardware as fh
from servepilot.testing.fake_engine import FakeEngine, FakeEngineBehavior
from tests.conftest import profile_from_config


def _engines() -> list[FakeEngine]:
    return [FakeEngine()]


def labels(result) -> list[str]:
    return [f"tp{p.tensor_parallel_size}x{p.replica_count}" for p in result.candidates]


class TestDivisors:
    def test_divisors(self) -> None:
        assert divisors(8) == [1, 2, 4, 8]
        assert divisors(6) == [1, 2, 3, 6]
        assert divisors(1) == [1]

    def test_heads_divisibility(self, dense_8b: ModelProfile) -> None:
        assert check_tp_divisibility(dense_8b, 8) is None
        assert check_tp_divisibility(dense_8b, 3) is not None


class TestGeneration:
    def test_single_gpu(self, dense_8b: ModelProfile, chat_workload: WorkloadProfile) -> None:
        result = generate_candidates(fh.h100x1(), dense_8b, chat_workload, _engines())
        assert labels(result) == ["tp1x1"]
        assert result.estimated_minimum_tp == 1
        plan = result.candidates[0]
        assert (
            plan.gpu_groups == [[0]]
            and plan.max_num_seqs is not None
            and plan.max_concurrency == plan.max_num_seqs
        )

    def test_two_gpus(self, dense_8b: ModelProfile, chat_workload: WorkloadProfile) -> None:
        result = generate_candidates(fh.h100x2(), dense_8b, chat_workload, _engines())
        assert labels(result) == ["tp1x2", "tp2x1"]

    def test_four_gpus(self, dense_8b: ModelProfile, chat_workload: WorkloadProfile) -> None:
        result = generate_candidates(fh.h100x4(), dense_8b, chat_workload, _engines())
        assert labels(result) == ["tp1x4", "tp2x2", "tp4x1"]
        tp2 = result.candidates[1]
        assert tp2.gpu_groups == [[0, 1], [2, 3]]

    def test_eight_gpus_small_model(
        self, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        result = generate_candidates(fh.h100x8(), dense_8b, chat_workload, _engines())
        assert labels(result) == ["tp1x8", "tp2x4", "tp4x2", "tp8x1"]
        assert all(p.viability == CandidateViability.ESTIMATED_VIABLE for p in result.candidates)
        assert [p.heuristic_rank for p in result.candidates] == [0, 1, 2, 3]

    def test_model_needing_tp2_prunes_tp1(
        self, dense_70b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        result = generate_candidates(fh.h100x4(), dense_70b, chat_workload, _engines())
        assert labels(result) == ["tp2x2", "tp4x1"]
        assert result.estimated_minimum_tp == 2
        excluded = [e for e in result.excluded if e.tensor_parallel_size == 1]
        assert excluded and "exceeds the safe budget" in excluded[0].reason

    def test_model_needing_all_gpus(self, chat_workload: WorkloadProfile) -> None:
        big = profile_from_config("llama3_70b", weight_bytes=int(240 * GIB))
        result = generate_candidates(fh.h100x4(), big, chat_workload, _engines())
        assert labels(result) == ["tp4x1"]

    def test_impossible_model(self, chat_workload: WorkloadProfile) -> None:
        huge = profile_from_config("llama3_70b", weight_bytes=int(700 * GIB))
        with pytest.raises(NoViablePlanError) as exc:
            generate_candidates(fh.h100x4(), huge, chat_workload, _engines())
        assert "quantized" in exc.value.render().lower()

    def test_latency_objective_ranks_large_tp_first(self, dense_8b: ModelProfile) -> None:
        wl = workload_from_preset("chat", objective=Objective.LATENCY, expected_concurrency=8)
        result = generate_candidates(fh.h100x4(), dense_8b, wl, _engines())
        assert labels(result) == ["tp4x1", "tp2x2", "tp1x4"]

    def test_context_capped_to_model_max(self, chat_workload: WorkloadProfile) -> None:
        gpt2 = profile_from_config("gpt2_small", weight_bytes=500 * 1024 * 1024)
        wl = workload_from_preset("chat")  # 8192 context > gpt2's 1024
        result = generate_candidates(fh.h100x1(), gpt2, wl, _engines())
        assert result.candidates[0].context_length == 1024
        assert any("exceeds the model's maximum" in n for n in result.notes)

    def test_explicit_context_beyond_model_requires_override(
        self, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        with pytest.raises(ConfigurationError):
            generate_candidates(
                fh.h100x1(),
                dense_8b,
                chat_workload,
                _engines(),
                PlanConstraints(context_length=16384),
            )
        result = generate_candidates(
            fh.h100x1(),
            dense_8b,
            chat_workload,
            _engines(),
            PlanConstraints(context_length=16384, allow_context_override=True),
        )
        assert result.candidates[0].context_length == 16384

    def test_dedupe(self, dense_8b: ModelProfile, chat_workload: WorkloadProfile) -> None:
        result = generate_candidates(fh.h100x4(), dense_8b, chat_workload, _engines())
        unique, dupes = dedupe(result.candidates + result.candidates)
        assert len(unique) == len(result.candidates) and len(dupes) == len(result.candidates)

    def test_unsupported_engine_excluded(
        self, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        engine = FakeEngine(FakeEngineBehavior(unsupported_model_types=["llama"]))
        with pytest.raises(NoViablePlanError):
            generate_candidates(fh.h100x1(), dense_8b, chat_workload, [engine])


class TestMoE:
    def test_ep_variants(self, moe_8x7b: ModelProfile, chat_workload: WorkloadProfile) -> None:
        result = generate_candidates(fh.h100x4(), moe_8x7b, chat_workload, _engines())
        ids = [p.id for p in result.candidates]
        assert "fake-tp2-x2" in ids and "fake-tp2-x2-ep" in ids and "fake-tp4-x1-ep" in ids
        ep = next(p for p in result.candidates if p.id == "fake-tp4-x1-ep")
        assert ep.expert_parallel_enabled and ep.expert_parallel_size == 4
        # Base variants rank before EP variants at the same TP.
        assert ids.index("fake-tp2-x2") < ids.index("fake-tp2-x2-ep")

    def test_ep_requires_divisible_experts(self, chat_workload: WorkloadProfile) -> None:
        moe = profile_from_config("mixtral_8x7b", weight_bytes=int(87 * GIB))
        moe.num_experts = 6
        result = generate_candidates(fh.h100x4(), moe, chat_workload, _engines())
        assert not any(
            p.expert_parallel_enabled and p.expert_parallel_size == 4 for p in result.candidates
        )
        assert any("not divisible by EP=4" in e.reason for e in result.excluded)


class TestOverrides:
    def test_tp_constraint(self, dense_8b: ModelProfile, chat_workload: WorkloadProfile) -> None:
        result = generate_candidates(
            fh.h100x8(),
            dense_8b,
            chat_workload,
            _engines(),
            PlanConstraints(tensor_parallel_size=2),
        )
        assert labels(result) == ["tp2x4"]

    def test_invalid_tp(self, dense_8b: ModelProfile, chat_workload: WorkloadProfile) -> None:
        with pytest.raises(ConfigurationError):
            generate_candidates(
                fh.h100x8(),
                dense_8b,
                chat_workload,
                _engines(),
                PlanConstraints(tensor_parallel_size=3),
            )

    def test_replicas_constraint(
        self, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        result = generate_candidates(
            fh.h100x8(),
            dense_8b,
            chat_workload,
            _engines(),
            PlanConstraints(tensor_parallel_size=2, replica_count=2),
        )
        assert labels(result) == ["tp2x2"]
        assert result.candidates[0].gpu_groups == [[0, 1], [2, 3]]
        with pytest.raises(ConfigurationError):
            generate_candidates(
                fh.h100x8(),
                dense_8b,
                chat_workload,
                _engines(),
                PlanConstraints(tensor_parallel_size=4, replica_count=3),
            )

    def test_gpu_subset(self, dense_8b: ModelProfile, chat_workload: WorkloadProfile) -> None:
        result = generate_candidates(
            fh.h100x8(), dense_8b, chat_workload, _engines(), PlanConstraints(gpu_ids=[4, 5])
        )
        assert result.selected_gpu_ids == [4, 5]
        assert labels(result) == ["tp1x2", "tp2x1"]
        assert result.candidates[1].gpu_groups == [[4, 5]]
        with pytest.raises(ConfigurationError):
            generate_candidates(
                fh.h100x8(), dense_8b, chat_workload, _engines(), PlanConstraints(gpu_ids=[0, 42])
            )

    def test_engine_constraint_and_max_concurrency(
        self, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        result = generate_candidates(
            fh.h100x2(),
            dense_8b,
            chat_workload,
            _engines(),
            PlanConstraints(engine=EngineName.FAKE, max_concurrency=12),
        )
        assert all(p.max_num_seqs == 12 for p in result.candidates)
        with pytest.raises(NoViablePlanError):
            generate_candidates(
                fh.h100x2(),
                dense_8b,
                chat_workload,
                _engines(),
                PlanConstraints(engine=EngineName.VLLM),
            )

    def test_memory_headroom_and_fraction(
        self, dense_32b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        loose = generate_candidates(
            fh.h100x1(), dense_32b, chat_workload, _engines(), PlanConstraints(memory_headroom=0.02)
        )
        assert loose.candidates[0].memory_fraction == pytest.approx(0.95)
        forced = generate_candidates(
            fh.h100x1(), dense_32b, chat_workload, _engines(), PlanConstraints(memory_fraction=0.9)
        )
        assert forced.candidates[0].memory_fraction == pytest.approx(0.9)


class TestHardwareEdgeCases:
    def test_heterogeneous_fails_clearly(
        self, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        with pytest.raises(HardwareError) as exc:
            generate_candidates(fh.mixed_h100_a100(), dense_8b, chat_workload, _engines())
        assert "--gpus 0,1" in exc.value.message and "--gpus 2,3" in exc.value.message

    def test_heterogeneous_subset_works(
        self, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        result = generate_candidates(
            fh.mixed_h100_a100(),
            dense_8b,
            chat_workload,
            _engines(),
            PlanConstraints(gpu_ids=[2, 3]),
        )
        assert result.selected_gpu_ids == [2, 3]

    def test_busy_gpu_warning(self, dense_8b: ModelProfile, chat_workload: WorkloadProfile) -> None:
        result = generate_candidates(
            fh.partially_occupied_x4(), dense_8b, chat_workload, _engines()
        )
        assert any("--allow-busy-gpus" in w for w in result.warnings)
        quiet = generate_candidates(
            fh.partially_occupied_x4(),
            dense_8b,
            chat_workload,
            _engines(),
            PlanConstraints(allow_busy_gpus=True),
        )
        assert all("--allow-busy-gpus" not in w for w in quiet.warnings)

    def test_no_gpus(self, dense_8b: ModelProfile, chat_workload: WorkloadProfile) -> None:
        snap = fh.h100x1().select([])
        with pytest.raises(HardwareError):
            generate_candidates(snap, dense_8b, chat_workload, _engines())

    def test_topology_unavailable_still_plans(
        self, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        result = generate_candidates(
            fh.topology_unavailable_x4(), dense_8b, chat_workload, _engines()
        )
        assert labels(result) == ["tp1x4", "tp2x2", "tp4x1"]
        assert result.candidates[1].gpu_groups == [[0, 1], [2, 3]]


class TestCluster:
    def test_per_node_replicas(
        self, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        result = generate_candidates(fh.cluster(2, 4), dense_8b, chat_workload, _engines())
        assert labels(result) == ["tp1x8", "tp2x4", "tp4x2"]
        tp4 = result.candidates[2]
        assert tp4.gpu_groups == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert tp4.replica_nodes == [["node-0"] * 4, ["node-1"] * 4]
        assert not tp4.spans_nodes

    def test_cross_node_when_model_exceeds_a_node(self, chat_workload: WorkloadProfile) -> None:
        huge = profile_from_config("llama3_70b", weight_bytes=int(400 * GIB))  # needs > 4 × 80 GB
        result = generate_candidates(fh.cluster(2, 4), huge, chat_workload, _engines())
        ids = [p.id for p in result.candidates]
        assert "fake-tp4-pp2-x1-ray" in ids
        plan = result.candidates[0]
        assert (
            plan.distributed_backend == "ray"
            and plan.spans_nodes
            and plan.pipeline_parallel_size == 2
        )
        assert plan.gpu_groups == [list(range(8))]

    def test_cross_node_needs_ray_capable_engine(self, chat_workload: WorkloadProfile) -> None:
        huge = profile_from_config("llama3_70b", weight_bytes=int(400 * GIB))

        class NoRay(FakeEngine):
            def supports_ray_backend(self) -> bool:
                return False

        with pytest.raises(NoViablePlanError):
            generate_candidates(fh.cluster(2, 4), huge, chat_workload, [NoRay()])


class TestPlannerFacade:
    def test_plan_and_heuristic_selection(
        self, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        planner = Planner(engines=_engines())
        result = planner.plan(fh.h100x4(), dense_8b, chat_workload)
        explanation = planner.explain(result, dense_8b, fh.h100x4())
        assert any("TP=1 is viable" in line for line in explanation)
        assert any("selected by benchmarking" in line for line in explanation)
        selected = Planner.heuristic_selection(result, chat_workload)
        assert not selected.benchmarked and selected.source == "heuristic"
        assert selected.plan.tensor_parallel_size == 1
        assert "UNBENCHMARKED" in selected.rationale[0]
