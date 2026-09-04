"""KV-cache estimation and the static memory model."""

from __future__ import annotations

import pytest

from servepilot.constants import GIB
from servepilot.models.kv_cache import (
    MLAKVEstimator,
    StandardAttentionKVEstimator,
    UnknownKVEstimator,
    estimate_kv,
    kv_element_bytes,
    select_kv_estimator,
)
from servepilot.planner.memory import (
    MemoryModelConfig,
    compute_memory_fraction,
    estimate_memory,
    format_bytes,
    minimum_tp_that_fits,
    safety_reserve_bytes,
    sharded_weight_bytes,
)
from servepilot.schemas.workload import workload_from_preset
from servepilot.testing import fake_hardware as fh
from tests.conftest import profile_from_config


class TestKV:
    def test_bf16_formula(self) -> None:
        p = profile_from_config("llama3_8b", weight_bytes=16 * GIB)
        est = estimate_kv(p, tp_size=1)
        # 2 × 32 layers × 8 kv heads × 128 head_dim × 2 bytes = 131072 B/token
        assert est.bytes_per_token_total == 131_072
        assert est.bytes_per_token_per_gpu == 131_072
        assert est.confidence == "high"
        assert "2 × 32 layers" in est.explanation
        # A profile without any head_dim information is only medium confidence.
        p.head_dim = None
        assert StandardAttentionKVEstimator().estimate(p, 1, None).confidence == "medium"

    def test_fp16_and_fp8_kv(self) -> None:
        assert kv_element_bytes(None, "float16") == 2
        assert kv_element_bytes("fp8", "bfloat16") == 1
        assert kv_element_bytes("auto", "bfloat16") == 2
        p = profile_from_config("qwen3_32b", weight_bytes=61 * GIB)
        full = estimate_kv(p, 1)
        half = estimate_kv(p, 1, kv_cache_dtype="fp8_e4m3")
        assert full.bytes_per_token_total == 2 * half.bytes_per_token_total  # type: ignore[operator]
        assert full.confidence == "high"

    def test_tp_shards_kv_heads_and_replicates_when_fewer_heads(self) -> None:
        p = profile_from_config("llama3_8b", weight_bytes=16 * GIB)  # 8 kv heads
        tp4 = estimate_kv(p, 4)
        tp16 = estimate_kv(p, 16)
        assert tp4.bytes_per_token_per_gpu == 131_072 // 4
        assert tp16.bytes_per_token_per_gpu == 131_072 // 8  # cannot go below one head per rank
        assert "replicated" in tp16.explanation

    def test_mla(self) -> None:
        p = profile_from_config("deepseek_v3", weight_bytes=642 * GIB)
        assert isinstance(select_kv_estimator(p), MLAKVEstimator)
        est = estimate_kv(p, 8)
        # 61 layers × (512 + 64) × 1 byte (fp8 weights → model dtype bf16 = 2 bytes)
        assert est.bytes_per_token_total == 61 * 576 * 2
        assert est.bytes_per_token_per_gpu == est.bytes_per_token_total
        assert est.confidence == "medium"

    def test_unknown_architectures(self) -> None:
        hybrid = profile_from_config("mamba_hybrid", weight_bytes=100 * GIB)
        assert isinstance(select_kv_estimator(hybrid), UnknownKVEstimator)
        est = estimate_kv(hybrid, 1)
        assert est.bytes_per_token_total is None and est.confidence == "unknown"
        missing = profile_from_config("missing_fields", weight_bytes=None)
        assert isinstance(select_kv_estimator(missing), UnknownKVEstimator)
        standard = StandardAttentionKVEstimator().estimate(missing, 1, None)
        assert standard.confidence == "unknown"


class TestMemoryModel:
    cfg = MemoryModelConfig()

    def test_reserve_and_fraction(self) -> None:
        total = 80 * GIB
        assert safety_reserve_bytes(total, self.cfg) == int(0.08 * total)
        assert compute_memory_fraction(total, total, self.cfg) == pytest.approx(0.92)
        # Half the memory occupied by another process → fraction derives from *free* memory.
        assert compute_memory_fraction(total, 40 * GIB, self.cfg) == pytest.approx(0.42)
        assert compute_memory_fraction(total, 0, self.cfg) == 0.0
        assert safety_reserve_bytes(8 * GIB, self.cfg) == GIB  # minimum absolute reserve

    def test_shard_overhead(self) -> None:
        assert sharded_weight_bytes(100, 1, self.cfg) == 100
        assert sharded_weight_bytes(1000, 2, self.cfg) == 505

    def test_small_model_fits_tp1(self) -> None:
        p = profile_from_config("llama3_8b", weight_bytes=16 * GIB)
        est = estimate_memory(
            p,
            gpus=fh.h100x1().gpus,
            tensor_parallel_size=1,
            kv=estimate_kv(p, 1),
            context_length=8192,
            workload=workload_from_preset("chat"),
            cfg=self.cfg,
        )
        assert est.fits and est.confidence == "high"
        assert est.weights_bytes == 16 * GIB
        assert est.kv_cache_bytes_available is not None and est.kv_cache_bytes_available > 40 * GIB
        assert (
            est.estimated_max_concurrency_p50 is not None
            and est.estimated_max_concurrency_p50 > 100
        )

    def test_large_model_needs_tp2(self) -> None:
        p = profile_from_config("llama3_70b", weight_bytes=131 * GIB)
        gpus = fh.h100x4().gpus
        wl = workload_from_preset("chat")
        tp1 = estimate_memory(
            p,
            gpus=gpus,
            tensor_parallel_size=1,
            kv=estimate_kv(p, 1),
            context_length=8192,
            workload=wl,
            cfg=self.cfg,
        )
        tp2 = estimate_memory(
            p,
            gpus=gpus,
            tensor_parallel_size=2,
            kv=estimate_kv(p, 2),
            context_length=8192,
            workload=wl,
            cfg=self.cfg,
        )
        assert not tp1.fits and tp1.shortfall_bytes > 50 * GIB
        assert tp2.fits
        assert (
            minimum_tp_that_fits(
                p,
                gpus=gpus,
                kv_for_tp=lambda tp: estimate_kv(p, tp),
                context_length=8192,
                workload=wl,
                cfg=self.cfg,
                candidates=[1, 2, 4],
            )
            == 2
        )

    def test_occupied_gpu_reduces_budget(self) -> None:
        p = profile_from_config("qwen3_32b", weight_bytes=61 * GIB)
        idle = fh.h100x4().gpus
        busy = fh.partially_occupied_x4().gpus  # GPU 1 has 30 GiB used
        wl = workload_from_preset("chat")
        ok = estimate_memory(
            p,
            gpus=idle,
            tensor_parallel_size=1,
            kv=estimate_kv(p, 1),
            context_length=8192,
            workload=wl,
            cfg=self.cfg,
        )
        constrained = estimate_memory(
            p,
            gpus=busy,
            tensor_parallel_size=1,
            kv=estimate_kv(p, 1),
            context_length=8192,
            workload=wl,
            cfg=self.cfg,
        )
        assert ok.fits
        assert not constrained.fits
        assert constrained.memory_fraction < ok.memory_fraction
        assert any("already in use" in n for n in constrained.notes)

    def test_unknown_kv_requires_spare(self) -> None:
        p = profile_from_config("mamba_hybrid", weight_bytes=66 * GIB)
        est = estimate_memory(
            p,
            gpus=fh.h100x1().gpus,
            tensor_parallel_size=1,
            kv=estimate_kv(p, 1),
            context_length=8192,
            workload=workload_from_preset("chat"),
            cfg=self.cfg,
        )
        assert est.kv_confidence == "unknown" and est.confidence == "low"
        assert est.estimated_max_concurrency_p50 is None
        assert est.fits  # 66 GiB weights + overheads leave > 2 GiB spare within the budget
        p2 = profile_from_config("mamba_hybrid", weight_bytes=72 * GIB)
        est2 = estimate_memory(
            p2,
            gpus=fh.h100x1().gpus,
            tensor_parallel_size=1,
            kv=estimate_kv(p2, 1),
            context_length=8192,
            workload=workload_from_preset("chat"),
            cfg=self.cfg,
        )
        assert not est2.fits

    def test_unknown_weights(self) -> None:
        p = profile_from_config("missing_fields", weight_bytes=None)
        est = estimate_memory(
            p,
            gpus=fh.h100x1().gpus,
            tensor_parallel_size=1,
            kv=estimate_kv(p, 1),
            context_length=2048,
            workload=workload_from_preset("chat", max_context_tokens=4096),
            cfg=self.cfg,
        )
        assert est.weights_bytes is None and est.confidence == "unknown" and est.fits

    def test_headroom_override(self) -> None:
        strict = MemoryModelConfig(headroom_fraction=0.30)
        assert compute_memory_fraction(80 * GIB, 80 * GIB, strict) == pytest.approx(0.70)

    def test_format_bytes(self) -> None:
        assert format_bytes(None) == "unknown"
        assert format_bytes(GIB) == "1.0 GiB"
        assert format_bytes(512 * 1024 * 1024) == "512 MiB"
