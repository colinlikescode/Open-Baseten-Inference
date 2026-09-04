"""Tuning cache: fingerprints, atomic store, schema versioning, validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from servepilot.cache.fingerprints import (
    engine_versions_compatible,
    hardware_fingerprint,
    model_fingerprint,
    validate_record,
    workload_fingerprint,
)
from servepilot.cache.store import CacheStore
from servepilot.constants import GIB
from servepilot.exceptions import CacheError
from servepilot.fsutil import atomic_write_json
from servepilot.schemas.benchmark import BenchmarkResult, BenchmarkSpec
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidateEvaluation, CandidatePlan, EngineName, SelectedPlan
from servepilot.schemas.runtime import TuningRecord
from servepilot.schemas.workload import (
    LatencyConstraints,
    Objective,
    WorkloadProfile,
    workload_from_preset,
)
from servepilot.testing import fake_hardware as fh


def make_plan(gpu_groups: list[list[int]] | None = None, **kw: object) -> CandidatePlan:
    return CandidatePlan(
        id="fake-tp1-x1",
        engine=EngineName.FAKE,
        gpu_groups=gpu_groups or [[0]],
        tensor_parallel_size=1,
        replica_count=len(gpu_groups or [[0]]),
        context_length=8192,
        memory_fraction=0.9,
        max_num_seqs=64,
        max_concurrency=64,
        **kw,  # type: ignore[arg-type]
    )


def make_result(cid: str = "fake-tp1-x1", tps: float = 1000.0) -> BenchmarkResult:
    return BenchmarkResult(
        candidate_id=cid,
        spec=BenchmarkSpec(
            concurrency=16,
            num_requests=32,
            seed=1,
            input_tokens_p50=1,
            input_tokens_p95=1,
            output_tokens_p50=1,
            output_tokens_p95=1,
        ),
        total_requests=32,
        successful_requests=32,
        failed_requests=0,
        duration_seconds=10,
        request_throughput=3.2,
        input_tokens_per_second=tps,
        output_tokens_per_second=tps,
        total_tokens_per_second=2 * tps,
        latency_p50_ms=100,
        latency_p95_ms=200,
        latency_p99_ms=300,
        error_rate=0.0,
    )


def make_record(
    model: ModelProfile,
    workload: WorkloadProfile,
    hardware=None,
    *,
    status: str = "complete",
    engine_versions: dict[str, str] | None = None,
) -> TuningRecord:
    hardware = hardware or fh.h100x1()
    plan = make_plan()
    winner = SelectedPlan(
        plan=plan, objective=workload.objective, benchmarked=True, final_result=make_result()
    )
    return TuningRecord(
        servepilot_version="1.0.0",
        hardware_fingerprint=hardware_fingerprint(hardware),
        model_fingerprint=model_fingerprint(model),
        workload_fingerprint=workload_fingerprint(workload),
        engine_versions=engine_versions if engine_versions is not None else {"fake": "0.0.1-fake"},
        hardware_snapshot=hardware,
        model_profile=model,
        workload_profile=workload,
        seed=1234,
        status=status,  # type: ignore[arg-type]
        candidates=[
            CandidateEvaluation(
                plan=plan, stage="structural", status="benchmarked", results=[make_result()]
            )
        ],
        winner=winner,
    )


class TestFingerprints:
    def test_workload_fingerprint_changes(self) -> None:
        chat = workload_from_preset("chat")
        assert workload_fingerprint(chat) == workload_fingerprint(workload_from_preset("chat"))
        assert workload_fingerprint(chat) != workload_fingerprint(
            workload_from_preset("long-context")
        )
        assert workload_fingerprint(chat) != workload_fingerprint(
            workload_from_preset("chat", objective=Objective.LATENCY)
        )
        assert workload_fingerprint(chat) != workload_fingerprint(
            workload_from_preset("chat", expected_concurrency=8)
        )
        assert workload_fingerprint(chat) != workload_fingerprint(
            workload_from_preset(
                "chat", latency_constraints=LatencyConstraints(max_p95_ttft_ms=500)
            )
        )
        assert workload_fingerprint(chat) != workload_fingerprint(
            workload_from_preset("chat", streaming=False)
        )

    def test_engine_version_compat(self) -> None:
        assert engine_versions_compatible("0.28.0", "0.28.3")
        assert not engine_versions_compatible("0.27.1", "0.28.0")
        assert not engine_versions_compatible(None, "0.28.0")


class TestStore:
    def test_roundtrip_list_clear(
        self, tmp_path: Path, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        store = CacheStore(tmp_path)
        record = make_record(dense_8b, chat_workload)
        path = store.save(record)
        assert path.exists() and path.parent == store.dir
        loaded = store.load(record.key)
        assert loaded.winner is not None and loaded.winner.plan.id == "fake-tp1-x1"
        assert loaded.model_profile.model_id == dense_8b.model_id
        found = store.find(
            record.hardware_fingerprint, record.model_fingerprint, record.workload_fingerprint
        )
        assert found is not None and found.key == record.key
        assert [r.key for r in store.list()] == [record.key]
        assert store.delete(record.key) and not store.delete(record.key)
        store.save(record)
        assert store.clear() == 1 and store.list() == []

    def test_missing_and_invalid(self, tmp_path: Path) -> None:
        store = CacheStore(tmp_path)
        with pytest.raises(CacheError):
            store.load("nope")
        store.dir.mkdir(parents=True)
        (store.dir / "bad.json").write_text("{not json")
        assert store.list() == []  # skipped with a warning
        with pytest.raises(CacheError):
            store._load_path(store.dir / "bad.json")

    def test_future_schema_rejected(
        self, tmp_path: Path, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        store = CacheStore(tmp_path)
        record = make_record(dense_8b, chat_workload)
        path = store.save(record)
        raw = json.loads(path.read_text())
        raw["schema_version"] = 99
        path.write_text(json.dumps(raw))
        with pytest.raises(CacheError) as exc:
            store.load(record.key)
        assert "schema version 99" in exc.value.message

    def test_atomic_write_leaves_no_temp_files(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "state.json"
        atomic_write_json(target, {"a": 1})
        atomic_write_json(target, {"a": 2})
        assert json.loads(target.read_text()) == {"a": 2}
        assert [p.name for p in target.parent.iterdir()] == ["state.json"]


class TestValidation:
    def test_valid_record(self, dense_8b: ModelProfile, chat_workload: WorkloadProfile) -> None:
        record = make_record(dense_8b, chat_workload)
        v = validate_record(
            record,
            hardware=fh.h100x1(),
            model=dense_8b,
            workload=chat_workload,
            engine_versions={"fake": "0.0.1-fake"},
        )
        assert v.valid, v.reasons

    def test_model_revision_change(
        self, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        record = make_record(dense_8b, chat_workload)
        changed = dense_8b.model_copy(update={"revision": "newsha"})
        v = validate_record(
            record,
            hardware=fh.h100x1(),
            model=changed,
            workload=chat_workload,
            engine_versions={"fake": "0.0.1-fake"},
        )
        assert not v.valid and any("model changed" in r for r in v.reasons)

    def test_workload_change(self, dense_8b: ModelProfile, chat_workload: WorkloadProfile) -> None:
        record = make_record(dense_8b, chat_workload)
        v = validate_record(
            record,
            hardware=fh.h100x1(),
            model=dense_8b,
            workload=workload_from_preset("long-context"),
            engine_versions={"fake": "0.0.1-fake"},
        )
        assert not v.valid and any("workload" in r for r in v.reasons)

    def test_topology_change(self, dense_8b: ModelProfile, chat_workload: WorkloadProfile) -> None:
        record = make_record(dense_8b, chat_workload, hardware=fh.h100x2())
        v = validate_record(
            record,
            hardware=fh.h100(2, nvlink=False),
            model=dense_8b,
            workload=chat_workload,
            engine_versions={"fake": "0.0.1-fake"},
        )
        assert not v.valid and any("hardware changed" in r for r in v.reasons)

    def test_engine_version_and_memory(
        self, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        record = make_record(dense_8b, chat_workload)
        v = validate_record(
            record,
            hardware=fh.h100x1(),
            model=dense_8b,
            workload=chat_workload,
            engine_versions={"fake": "0.1.0"},
        )
        assert any("version changed" in r for r in v.reasons)
        missing = validate_record(
            record, hardware=fh.h100x1(), model=dense_8b, workload=chat_workload, engine_versions={}
        )
        assert any("no longer available" in r for r in missing.reasons)
        busy = fh.h100x1()
        busy.gpus[0].free_memory_bytes = 10 * GIB
        v2 = validate_record(
            record,
            hardware=busy,
            model=dense_8b,
            workload=chat_workload,
            engine_versions={"fake": "0.0.1-fake"},
        )
        assert any("free but the cached plan needs" in r for r in v2.reasons)

    def test_incomplete_record(
        self, dense_8b: ModelProfile, chat_workload: WorkloadProfile
    ) -> None:
        record = make_record(dense_8b, chat_workload, status="in_progress")
        v = validate_record(
            record,
            hardware=fh.h100x1(),
            model=dense_8b,
            workload=chat_workload,
            engine_versions={"fake": "0.0.1-fake"},
        )
        assert any("incomplete" in r for r in v.reasons)
