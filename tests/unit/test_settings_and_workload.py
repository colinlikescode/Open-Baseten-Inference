"""Configuration loading/merging, workload presets, prompt generation and metric aggregation."""

from __future__ import annotations

from pathlib import Path

import pytest

from servepilot.benchmark.metrics import aggregate, aggregate_gpu_samples
from servepilot.benchmark.runner import make_spec
from servepilot.benchmark.workload import PromptGenerator, warmup_count
from servepilot.cli.common import PlanFlags, build_config, constraints_from_config
from servepilot.exceptions import ConfigurationError
from servepilot.models.tokenizer import ApproximateTokenizer
from servepilot.schemas.benchmark import BenchmarkSpec, RequestBenchmarkResult
from servepilot.schemas.hardware import GPUSample
from servepilot.schemas.plan import EngineName
from servepilot.schemas.workload import Objective, WorkloadProfile, workload_from_preset
from servepilot.settings import ServePilotConfig, load_config


class TestWorkloadPresets:
    def test_presets(self) -> None:
        chat = workload_from_preset("chat")
        assert (
            chat.input_tokens_p50,
            chat.input_tokens_p95,
            chat.output_tokens_p50,
            chat.output_tokens_p95,
        ) == (512, 2048, 256, 768)
        lc = workload_from_preset("long-context")
        assert lc.input_tokens_p95 == 32768 and lc.max_context_tokens >= lc.p95_sequence_tokens
        dh = workload_from_preset("decode-heavy")
        assert dh.output_tokens_p95 == 4096
        with pytest.raises(ValueError, match="unknown workload preset"):
            workload_from_preset("nope")

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="exceeds max_context_tokens"):
            WorkloadProfile(input_tokens_p95=8000, output_tokens_p95=1000, max_context_tokens=8192)
        with pytest.raises(ValueError, match="must be >="):
            WorkloadProfile(input_tokens_p50=1000, input_tokens_p95=500)


class TestConfig:
    def test_yaml_load_and_merge(self, tmp_path: Path) -> None:
        cfg = tmp_path / "servepilot.yaml"
        cfg.write_text(
            """
model: Qwen/Qwen3-32B
engine: sglang
objective: balanced
profile:
  name: long-context
  expected_concurrency: 32
  max_p95_ttft_ms: 800
hardware:
  gpus: [0, 1]
  memory_headroom: 0.1
server:
  host: 0.0.0.0
  port: 9000
tuning:
  enabled: true
  use_cache: false
  stage_a_requests: 8
constraints:
  tp: 2
  engine_args:
    enable-prefix-caching: true
"""
        )
        config = load_config(cfg)
        assert (
            config.model == "Qwen/Qwen3-32B"
            and config.engine == "sglang"
            and config.objective == Objective.BALANCED
        )
        assert (
            config.hardware.gpus == [0, 1]
            and config.server.port == 9000
            and config.tuning.stage_a_requests == 8
        )
        merged = config.merged({"server.port": 8001, "model": None, "constraints.tp": 4})
        assert (
            merged.server.port == 8001
            and merged.model == "Qwen/Qwen3-32B"
            and merged.constraints.tp == 4
        )
        workload = merged.build_workload()
        assert (
            workload.name == "long-context"
            and workload.expected_concurrency == 32
            and workload.objective == Objective.BALANCED
        )
        assert (
            workload.latency_constraints is not None
            and workload.latency_constraints.max_p95_ttft_ms == 800
        )
        constraints = constraints_from_config(merged)
        assert (
            constraints.engine == EngineName.SGLANG
            and constraints.gpu_ids == [0, 1]
            and constraints.tensor_parallel_size == 4
        )
        assert (
            constraints.extra_engine_args == {"enable-prefix-caching": True}
            and constraints.memory_headroom == 0.1
        )

    def test_unknown_keys_and_bad_values(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("model: x\nbogus: 1\n")
        with pytest.raises(ConfigurationError) as exc:
            load_config(bad)
        assert "bogus" in exc.value.message
        bad.write_text("model: x\nserver:\n  port: 70000\n")
        with pytest.raises(ConfigurationError):
            load_config(bad)
        bad.write_text("- not a mapping\n")
        with pytest.raises(ConfigurationError):
            load_config(bad)
        bad.write_text("model: [unclosed\n")
        with pytest.raises(ConfigurationError):
            load_config(bad)
        with pytest.raises(ConfigurationError):
            load_config(tmp_path / "missing.yaml")
        assert load_config(None) == ServePilotConfig()

    def test_flags_override_and_custom_profile(self) -> None:
        flags = PlanFlags(
            model="m",
            input_tokens=1500,
            output_tokens=500,
            context_length=8192,
            objective="throughput",
            gpus="0,2",
            tp=2,
            no_stream=True,
        )
        config = build_config(flags)
        wl = config.build_workload()
        assert wl.name == "custom" and wl.input_tokens_p50 == 1500 and wl.input_tokens_p95 == 6000
        assert (
            wl.output_tokens_p50 == 500
            and wl.output_tokens_p95 == 1500
            and wl.max_context_tokens == 8192
        )
        assert wl.streaming is False
        assert config.hardware.gpus == [0, 2] and config.constraints.tp == 2
        # A p50 override that would exceed the preset context grows the context automatically.
        grown = build_config(PlanFlags(model="m", input_tokens=4000)).build_workload()
        assert grown.max_context_tokens >= grown.input_tokens_p95 + grown.output_tokens_p95

    def test_flag_validation(self) -> None:
        with pytest.raises(ConfigurationError):
            build_config(PlanFlags(model=None))
        with pytest.raises(ConfigurationError):
            build_config(PlanFlags(model="m", gpus="a,b"))
        with pytest.raises(ConfigurationError):
            build_config(PlanFlags(model="m", engine="tensorrt"))
        with pytest.raises(ConfigurationError):
            build_config(PlanFlags(model="m", objective="speed"))
        with pytest.raises(ConfigurationError):
            build_config(PlanFlags(model="m", profile="nope"))


class TestPromptGeneration:
    def test_deterministic_and_length_targets(self) -> None:
        tok = ApproximateTokenizer()
        wl = workload_from_preset("chat")
        gen_a = PromptGenerator(tok, seed=7).generate(wl, 40)
        gen_b = PromptGenerator(tok, seed=7).generate(wl, 40)
        assert [r.prompt for r in gen_a] == [r.prompt for r in gen_b]
        assert len({r.prompt for r in gen_a}) > 30, "prompts must not be a repeated tiny string"
        # p95 requests appear at every 20th position.
        assert (
            gen_a[19].input_tokens >= wl.input_tokens_p95 * 0.9
            and gen_a[19].max_tokens == wl.output_tokens_p95
        )
        typical = [r.input_tokens for i, r in enumerate(gen_a) if i % 20 != 19]
        assert (
            min(typical) >= wl.input_tokens_p50 * 0.6 and max(typical) <= wl.input_tokens_p95 * 1.15
        )
        assert all(r.input_tokens + r.max_tokens <= wl.max_context_tokens for r in gen_a)
        different = PromptGenerator(tok, seed=8).generate(wl, 5)
        assert different[0].prompt != gen_a[0].prompt

    def test_shared_prefix(self) -> None:
        wl = workload_from_preset("chat").model_copy(update={"shared_prefix_fraction": 0.5})
        reqs = PromptGenerator(ApproximateTokenizer(), seed=1).generate(wl, 5)
        assert (
            all(r.prompt.startswith(reqs[0].shared_prefix) for r in reqs) and reqs[0].shared_prefix
        )

    def test_warmup_count(self) -> None:
        assert warmup_count(1, 2, 16) == 2
        assert warmup_count(16, 2, 16) == 8
        assert warmup_count(1024, 2, 16) == 16

    def test_make_spec(self) -> None:
        spec = make_spec(
            workload_from_preset("chat"), concurrency=64, num_requests=10, seed=3, request_rate=2.0
        )
        assert spec.mode == "open" and spec.warmup_requests == 16 and spec.streaming


class TestAggregation:
    def test_throughput_uses_wall_clock_not_mean_latency(self) -> None:
        spec = BenchmarkSpec(
            concurrency=2,
            num_requests=4,
            seed=1,
            input_tokens_p50=1,
            input_tokens_p95=1,
            output_tokens_p50=1,
            output_tokens_p95=1,
        )
        results = [
            RequestBenchmarkResult(
                success=True,
                input_tokens=10,
                output_tokens=100,
                started_at=0.0,
                first_token_at=0.1,
                completed_at=1.0,
                ttft_ms=100,
                e2e_latency_ms=1000,
                tpot_ms=9.09,
            ),
            RequestBenchmarkResult(
                success=True,
                input_tokens=10,
                output_tokens=100,
                started_at=0.0,
                first_token_at=0.2,
                completed_at=2.0,
                ttft_ms=200,
                e2e_latency_ms=2000,
                tpot_ms=18.2,
            ),
            RequestBenchmarkResult(
                success=False, input_tokens=10, started_at=0.5, error="HTTP 500"
            ),
            RequestBenchmarkResult(
                success=True,
                input_tokens=10,
                output_tokens=200,
                started_at=1.0,
                first_token_at=1.1,
                completed_at=4.0,
                ttft_ms=100,
                e2e_latency_ms=3000,
                tpot_ms=14.6,
            ),
        ]
        agg = aggregate(
            "c",
            spec,
            results,
            gpu_samples=[
                GPUSample(
                    index=0,
                    timestamp=0,
                    utilization_percent=50,
                    memory_used_bytes=10,
                    power_watts=100,
                ),
                GPUSample(
                    index=0,
                    timestamp=1,
                    utilization_percent=90,
                    memory_used_bytes=20,
                    power_watts=300,
                ),
            ],
        )
        assert agg.total_requests == 4 and agg.successful_requests == 3 and agg.failed_requests == 1
        assert agg.duration_seconds == pytest.approx(4.0)
        assert agg.output_tokens_per_second == pytest.approx(400 / 4.0)
        assert agg.request_throughput == pytest.approx(3 / 4.0)
        assert agg.error_rate == pytest.approx(0.25)
        assert agg.latency_p50_ms == 2000 and agg.ttft_p95_ms == pytest.approx(190.0)
        assert (
            agg.gpu_metrics.mean_gpu_utilization == 70
            and agg.gpu_metrics.peak_memory_bytes == 20
            and agg.gpu_metrics.mean_power_watts == 200
        )
        assert agg.errors_sample == ["HTTP 500"]
        assert "c=2" in agg.short()
        assert aggregate_gpu_samples([]).sample_count == 0

    def test_empty(self) -> None:
        spec = BenchmarkSpec(
            concurrency=1,
            num_requests=1,
            seed=1,
            input_tokens_p50=1,
            input_tokens_p95=1,
            output_tokens_p50=1,
            output_tokens_p95=1,
        )
        agg = aggregate("c", spec, [])
        assert agg.error_rate == 1.0 and agg.output_tokens_per_second == 0
