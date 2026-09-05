"""Engine adapters: launch specs, support decisions, version gating, registry, interpreter discovery."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from servepilot.constants import GIB
from servepilot.engines.base import redacted_command
from servepilot.engines.interpreter import EngineRuntime, find_engine_runtime, parse_version
from servepilot.engines.registry import EngineRegistry, build_registry, select_engines
from servepilot.engines.sglang import SGLangEngine
from servepilot.engines.vllm import VLLMEngine
from servepilot.exceptions import EngineUnavailableError
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidatePlan, EngineName, MemoryEstimate
from servepilot.schemas.workload import Objective
from servepilot.settings import ServePilotSettings
from servepilot.testing.fake_engine import FakeEngine, FakeEngineBehavior
from tests.conftest import profile_from_config

RUNTIME = EngineRuntime(python="/opt/venv/bin/python", version="0.28.0")


def plan(**kw: object) -> CandidatePlan:
    base: dict[str, object] = {
        "id": "vllm-tp2-x1",
        "engine": EngineName.VLLM,
        "gpu_groups": [[0, 1]],
        "tensor_parallel_size": 2,
        "replica_count": 1,
        "context_length": 8192,
        "memory_fraction": 0.9,
        "max_num_seqs": 128,
        "max_running_requests": 128,
    }
    base.update(kw)
    return CandidatePlan(**base)  # type: ignore[arg-type]


class TestVersion:
    def test_parse(self) -> None:
        assert parse_version("0.28.0") == (0, 28, 0)
        assert parse_version("0.6.4.post1") == (0, 6, 4)
        assert parse_version("1.0rc1") == (1, 0)
        assert parse_version(None) == () and parse_version("garbage") == ()


class TestVLLM:
    def test_launch_spec(self, dense_8b: ModelProfile, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_secret123456789")
        monkeypatch.setenv("UNRELATED", "x")
        engine = VLLMEngine(RUNTIME)
        dense_8b.revision = "abc"
        spec = engine.build_launch_spec(
            dense_8b,
            plan(),
            replica_index=0,
            host="127.0.0.1",
            port=30001,
            served_model_name="llama",
            trust_remote_code=True,
        )
        assert spec.executable == "/opt/venv/bin/python"
        args = spec.args
        assert args[:3] == ["-m", "vllm.entrypoints.openai.api_server", "--model"]
        for flag, value in [
            ("--tensor-parallel-size", "2"),
            ("--gpu-memory-utilization", "0.90"),
            ("--max-model-len", "8192"),
            ("--max-num-seqs", "128"),
            ("--dtype", "bfloat16"),
            ("--revision", "abc"),
            ("--served-model-name", "llama"),
            ("--port", "30001"),
        ]:
            assert args[args.index(flag) + 1] == value
        assert "--trust-remote-code" in args and "--disable-uvicorn-access-log" in args
        assert (
            spec.env["CUDA_VISIBLE_DEVICES"] == "0,1"
            and spec.env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        )
        assert spec.env["HF_TOKEN"] == "hf_secret123456789" and "UNRELATED" not in spec.env
        assert (
            "hf_secret" not in spec.redacted_display_command
            and "HF_TOKEN=***" in spec.redacted_display_command
        )
        assert spec.base_url == "http://127.0.0.1:30001" and spec.replica_id == "vllm-tp2-x1-r0"

    def test_moe_ep_pp_ray_and_extra_args(self, moe_8x7b: ModelProfile) -> None:
        engine = VLLMEngine(RUNTIME)
        p = plan(
            expert_parallel_enabled=True,
            expert_parallel_size=2,
            pipeline_parallel_size=2,
            distributed_backend="ray",
            replica_nodes=[["n0", "n0", "n1", "n1"]],
            gpu_groups=[[0, 1, 2, 3]],
            kv_cache_dtype="fp8",
            chunked_prefill_enabled=False,
            engine_args={"enable-prefix-caching": True, "seed": 7, "skip": False},
        )
        spec = engine.build_launch_spec(
            moe_8x7b, p, replica_index=0, host="0.0.0.0", port=1, node=("n0", "10.0.0.1")
        )
        args = spec.args
        assert "--enable-expert-parallel" in args
        assert args[args.index("--pipeline-parallel-size") + 1] == "2"
        assert args[args.index("--distributed-executor-backend") + 1] == "ray"
        assert args[args.index("--kv-cache-dtype") + 1] == "fp8"
        assert "--no-enable-chunked-prefill" in args
        assert (
            "--enable-prefix-caching" in args
            and args[args.index("--seed") + 1] == "7"
            and "--skip" not in args
        )
        assert "CUDA_VISIBLE_DEVICES" not in spec.env  # spans nodes: Ray assigns GPUs
        assert spec.base_url == "http://10.0.0.1:1"

    def test_old_version_flags(self, dense_8b: ModelProfile) -> None:
        old = VLLMEngine(EngineRuntime("/py", "0.6.0"))
        spec = old.build_launch_spec(
            dense_8b, plan(chunked_prefill_enabled=False), replica_index=0, host="h", port=1
        )
        assert (
            "--enable-chunked-prefill=False" in spec.args
            and "--disable-uvicorn-access-log" not in spec.args
        )
        ancient = VLLMEngine(EngineRuntime("/py", "0.4.0"))
        assert not ancient.supports(dense_8b, plan()).supported

    def test_support_decisions(
        self, dense_8b: ModelProfile, moe_8x7b: ModelProfile, deepseek_v3: ModelProfile
    ) -> None:
        engine = VLLMEngine(RUNTIME)
        ok = engine.supports(dense_8b, plan())
        assert ok.supported and ok.confidence == "high"
        assert engine.supports_expert_parallel(moe_8x7b) and engine.supports_dp_attention(moe_8x7b)
        assert not engine.supports_dp_attention(dense_8b)
        dpa = engine.supports(dense_8b, plan(dp_attention_enabled=True))
        assert not dpa.supported and "DP attention" in dpa.reasons[0]
        # DP attention needs the single-process DP mode that arrived in 0.9.
        old = VLLMEngine(EngineRuntime("/py", "0.8.5"))
        assert old.supports_expert_parallel(moe_8x7b) and not old.supports_dp_attention(moe_8x7b)
        weird = profile_from_config("llama3_70b_awq", weight_bytes=37 * GIB)
        weird.quantization_config = {"quant_method": "exotic-quant"}
        assert not engine.supports(weird, plan()).supported
        ds = engine.supports(deepseek_v3, plan())
        assert ds.supported and any("trust_remote_code" in w for w in ds.warnings)

    def test_dp_attention_launch_spec(self, moe_8x7b: ModelProfile) -> None:
        engine = VLLMEngine(RUNTIME)
        p = plan(
            id="vllm-tp4-x1-dpa",
            gpu_groups=[[0, 1, 2, 3]],
            tensor_parallel_size=4,
            data_parallel_size=4,
            expert_parallel_enabled=True,
            expert_parallel_size=4,
            dp_attention_enabled=True,
            max_num_seqs=1000,
        )
        args = engine.build_launch_spec(moe_8x7b, p, replica_index=0, host="h", port=1).args
        # vLLM runs DP attention as TP=1 × DP=N with expert parallelism on the same GPUs.
        assert args[args.index("--tensor-parallel-size") + 1] == "1"
        assert args[args.index("--data-parallel-size") + 1] == "4"
        assert "--enable-expert-parallel" in args
        # --max-num-seqs is per DP rank; the plan's limit is for the whole replica.
        assert args[args.index("--max-num-seqs") + 1] == "250"

    def test_performance_mode_follows_objective(self, dense_8b: ModelProfile) -> None:
        engine = VLLMEngine(RUNTIME)

        def mode(**kw: object) -> str | None:
            args = engine.build_launch_spec(
                dense_8b, plan(**kw), replica_index=0, host="h", port=1
            ).args
            return (
                args[args.index("--performance-mode") + 1] if "--performance-mode" in args else None
            )

        assert mode(objective=Objective.THROUGHPUT) == "throughput"
        assert mode(objective=Objective.LATENCY) == "interactivity"
        assert mode(objective=Objective.BALANCED) is None  # vLLM's default
        assert mode() is None
        old = VLLMEngine(EngineRuntime("/py", "0.16.0"))
        spec = old.build_launch_spec(
            dense_8b, plan(objective=Objective.THROUGHPUT), replica_index=0, host="h", port=1
        )
        assert "--performance-mode" not in spec.args

    def test_unavailable(self) -> None:
        engine = VLLMEngine(None, probe=False)
        assert not engine.is_available() and engine.version() is None
        assert any("SERVEPILOT_VLLM_PYTHON" in h for h in engine.unavailable_hints())
        with pytest.raises(RuntimeError):
            engine.build_launch_spec(
                profile_from_config("llama3_8b", weight_bytes=1),
                plan(),
                replica_index=0,
                host="h",
                port=1,
            )


class TestSGLang:
    def test_launch_spec(self, moe_8x7b: ModelProfile) -> None:
        engine = SGLangEngine(EngineRuntime("/sg/python", "0.5.18"))
        p = plan(
            engine=EngineName.SGLANG,
            expert_parallel_enabled=True,
            expert_parallel_size=2,
            data_parallel_size=2,
            dp_attention_enabled=True,
            chunked_prefill_size=4096,
            chunked_prefill_enabled=True,
        )
        spec = engine.build_launch_spec(
            moe_8x7b, p, replica_index=0, host="127.0.0.1", port=30002, trust_remote_code=False
        )
        args = spec.args
        assert args[:3] == ["-m", "sglang.launch_server", "--model-path"]
        for flag, value in [
            ("--tp-size", "2"),
            ("--dp-size", "2"),
            ("--ep-size", "2"),
            ("--mem-fraction-static", "0.90"),
            ("--context-length", "8192"),
            ("--max-running-requests", "128"),
            ("--chunked-prefill-size", "4096"),
            ("--log-level", "warning"),
        ]:
            assert args[args.index(flag) + 1] == value
        assert "--enable-dp-attention" in args and "--trust-remote-code" not in args
        assert spec.env["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_support(
        self, moe_8x7b: ModelProfile, deepseek_v3: ModelProfile, dense_8b: ModelProfile
    ) -> None:
        engine = SGLangEngine(EngineRuntime("/sg/python", "0.5.18"))
        assert engine.supports_dp_attention(deepseek_v3) and not engine.supports_dp_attention(
            dense_8b
        )
        res = engine.supports(moe_8x7b, plan(engine=EngineName.SGLANG, dp_attention_enabled=True))
        assert not res.supported and "DP attention" in res.reasons[0]
        ray = engine.supports(dense_8b, plan(engine=EngineName.SGLANG, distributed_backend="ray"))
        assert not ray.supported and not engine.supports_ray_backend()
        assert engine.supports(dense_8b, plan(engine=EngineName.SGLANG)).supported
        disabled = engine.build_launch_spec(
            dense_8b,
            plan(engine=EngineName.SGLANG, chunked_prefill_enabled=False),
            replica_index=0,
            host="h",
            port=1,
        )
        assert disabled.args[disabled.args.index("--chunked-prefill-size") + 1] == "-1"

    def test_memory_fraction_and_cuda_graphs_from_estimate(self, dense_8b: ModelProfile) -> None:
        """SGLang's static fraction excludes activations/graphs; graph capture follows the limit."""
        engine = SGLangEngine(EngineRuntime("/sg/python", "0.5.18"))
        estimate = MemoryEstimate(
            device_total_bytes=80 * GIB,
            device_free_bytes=79 * GIB,
            memory_fraction=0.91,
            engine_budget_bytes=int(0.91 * 80 * GIB),
            weights_bytes=16 * GIB,
            activations_bytes=2 * GIB,
            cuda_graph_bytes=1 * GIB,
            communication_bytes=0,
            engine_overhead_bytes=int(1.5 * GIB),
            safety_reserve_bytes=6 * GIB,
            kv_cache_bytes_available=50 * GIB,
            kv_bytes_per_token_per_gpu=131072,
            fits=True,
        )
        p = plan(
            engine=EngineName.SGLANG,
            tensor_parallel_size=1,
            gpu_groups=[[0]],
            memory_fraction=0.91,
            max_num_seqs=512,
            max_running_requests=512,
            estimated_memory=estimate,
        )
        args = engine.build_launch_spec(dense_8b, p, replica_index=0, host="h", port=1).args
        # 0.91 − (2 + 1 + 1.5) GiB / 80 GiB ≈ 0.85
        assert args[args.index("--mem-fraction-static") + 1] == "0.85"
        assert args[args.index("--cuda-graph-max-bs-decode") + 1] == "512"
        older = SGLangEngine(EngineRuntime("/sg/python", "0.4.9"))
        old_args = older.build_launch_spec(dense_8b, p, replica_index=0, host="h", port=1).args
        assert old_args[old_args.index("--cuda-graph-max-bs") + 1] == "512"
        # Within SGLang's own default capture range: leave the flag out.
        small = p.with_updates(max_num_seqs=200, max_running_requests=200)
        small_args = engine.build_launch_spec(
            dense_8b, small, replica_index=0, host="h", port=1
        ).args
        assert not any(a.startswith("--cuda-graph-max-bs") for a in small_args)
        # Without an estimate the fraction passes through untouched.
        bare = engine.build_launch_spec(
            dense_8b, plan(engine=EngineName.SGLANG), replica_index=0, host="h", port=1
        ).args
        assert bare[bare.index("--mem-fraction-static") + 1] == "0.90"


class TestRegistry:
    def test_registry_and_selection(self, settings: ServePilotSettings) -> None:
        fake = FakeEngine()
        registry = EngineRegistry(
            [VLLMEngine(None, probe=False), SGLangEngine(None, probe=False), fake]
        )
        assert registry.available() == [fake]
        assert registry.versions() == {"fake": "0.0.1-fake"}
        assert select_engines(registry, "auto") == [
            fake
        ]  # fake only when nothing else is installed
        assert select_engines(registry, "fake") == [fake]
        with pytest.raises(EngineUnavailableError) as exc:
            select_engines(registry, "vllm")
        assert "not installed" in exc.value.message
        with pytest.raises(EngineUnavailableError):
            registry.get("nope")
        empty = EngineRegistry([VLLMEngine(None, probe=False)])
        with pytest.raises(EngineUnavailableError):
            select_engines(empty, "auto")

    def test_auto_prefers_real_engines(self) -> None:
        real = VLLMEngine(RUNTIME)
        registry = EngineRegistry([real, FakeEngine()])
        assert select_engines(registry, "auto") == [real]

    def test_build_registry_respects_fake_flag(
        self, settings: ServePilotSettings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SERVEPILOT_VLLM_PYTHON", "/definitely/missing/python")
        names = {e.name() for e in build_registry(settings).all()}
        assert names == {"vllm", "sglang"}
        with_fake = build_registry(ServePilotSettings(enable_fake_engine=True))
        assert "fake" in {e.name() for e in with_fake.all()}


class TestInterpreterDiscovery:
    def test_explicit_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Point at the current interpreter; the distribution must resolve (pytest is installed).
        import sys

        monkeypatch.setenv("SERVEPILOT_TEST_PYTHON", sys.executable)
        rt = find_engine_runtime("pytest", env_var="SERVEPILOT_TEST_PYTHON")
        assert rt is not None and rt.python == sys.executable and rt.version
        monkeypatch.setenv("SERVEPILOT_TEST_PYTHON", str(tmp_path / "nope"))
        assert find_engine_runtime("pytest", env_var="SERVEPILOT_TEST_PYTHON") is None

    def test_current_interpreter(self) -> None:
        rt = find_engine_runtime("pytest", env_var="SERVEPILOT_UNSET_VAR_XYZ")
        assert rt is not None and rt.version
        assert (
            find_engine_runtime("definitely-not-installed-dist", env_var="SERVEPILOT_UNSET_VAR_XYZ")
            is None
        )

    def test_shebang_console_script(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import sys

        script = tmp_path / "pytestscript"
        script.write_text(f"#!{sys.executable}\nprint('hi')\n")
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
        monkeypatch.setattr(
            "servepilot.engines.interpreter.metadata.version",
            lambda d: (_ for _ in ()).throw(
                __import__("importlib.metadata").metadata.PackageNotFoundError(d)
            ),
        )
        rt = find_engine_runtime(
            "pytest", env_var="SERVEPILOT_UNSET_VAR_XYZ", console_scripts=("pytestscript",)
        )
        assert rt is not None and rt.python == sys.executable

    def test_shebang_parsing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from servepilot.engines.interpreter import _shebang_interpreter

        bare = tmp_path / "bare"
        bare.write_text("#!\n")
        assert _shebang_interpreter(bare) is None
        plain = tmp_path / "plain"
        plain.write_text("print(1)\n")
        assert _shebang_interpreter(plain) is None
        # `#!/usr/bin/env NAME` resolves NAME through PATH.
        bindir = tmp_path / "bin"
        bindir.mkdir()
        interp = bindir / "mypython"
        interp.symlink_to(sys.executable)
        monkeypatch.setenv("PATH", str(bindir))
        via_env = tmp_path / "via_env"
        via_env.write_text("#!/usr/bin/env mypython\n")
        assert _shebang_interpreter(via_env) == str(interp)
        missing = tmp_path / "missing"
        missing.write_text("#!/definitely/not/here\n")
        assert _shebang_interpreter(missing) is None


def test_redaction_helper() -> None:
    cmd = redacted_command(
        ["python", "-m", "x"],
        {"HF_TOKEN": "hf_abc", "CUDA_VISIBLE_DEVICES": "0", "VLLM_FOO": "1", "PATH": "/bin"},
    )
    assert (
        "HF_TOKEN=***" in cmd
        and "CUDA_VISIBLE_DEVICES=0" in cmd
        and "VLLM_FOO=1" in cmd
        and "PATH" not in cmd
    )


def test_fake_engine_behaviour_lookup(dense_8b: ModelProfile) -> None:
    behavior = FakeEngineBehavior(
        plans={"tp2": {"startup": "oom"}, "fake-tp1-x4": {"tpot_ms": 1.0}}
    )  # type: ignore[dict-item]
    engine = FakeEngine(behavior)
    p2 = plan(
        engine=EngineName.FAKE, id="fake-tp2-x2", replica_count=2, gpu_groups=[[0, 1], [2, 3]]
    )
    assert behavior.for_plan(p2).startup == "oom"
    p1 = plan(
        engine=EngineName.FAKE,
        id="fake-tp1-x4",
        tensor_parallel_size=1,
        replica_count=4,
        gpu_groups=[[0], [1], [2], [3]],
    )
    _ttft, tpot, capacity = behavior.performance(p1)
    assert tpot == 1.0 and capacity == behavior.base_capacity
    spec = engine.build_launch_spec(dense_8b, p2, replica_index=1, host="127.0.0.1", port=5)
    assert (
        "--startup-mode" in spec.args and spec.args[spec.args.index("--startup-mode") + 1] == "oom"
    )
    assert spec.env["CUDA_VISIBLE_DEVICES"] == "2,3"
