"""Model inspection: config normalisation, weight sizing, Hub metadata handling, fingerprints."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from servepilot.constants import GIB
from servepilot.exceptions import ModelInspectionError
from servepilot.models.fingerprint import config_hash, model_fingerprint
from servepilot.models.inspector import ModelInspector, inspect_local_model, normalize_config
from servepilot.models.weights import (
    bytes_from_safetensors_parameter_counts,
    estimate_parameter_count,
    estimate_weight_bytes,
    measure_hub_weights,
    measure_local_weights,
    quantized_bytes_per_parameter,
    select_weight_files,
)
from servepilot.schemas.model import AttentionKind
from tests.conftest import load_config, make_local_model, profile_from_config


class TestNormalize:
    def test_dense_gqa(self) -> None:
        p = normalize_config("llama", load_config("llama3_8b"))
        assert p.attention_kind == AttentionKind.GQA
        assert p.head_dim == 128 and p.num_key_value_heads == 8
        assert not p.is_moe and not p.is_multimodal and not p.trust_remote_code_required
        assert p.configured_dtype == "bfloat16"
        assert p.estimated_parameter_count is not None
        assert 7.5e9 < p.estimated_parameter_count < 8.5e9

    def test_mha_and_gpt2_aliases(self) -> None:
        p = normalize_config("gpt2", load_config("gpt2_small"))
        assert p.hidden_size == 768 and p.num_hidden_layers == 12 and p.num_attention_heads == 12
        assert p.attention_kind == AttentionKind.MHA
        assert p.max_position_embeddings == 1024
        assert p.hidden_act == "gelu_new"
        assert (
            p.estimated_parameter_count is not None and 100e6 < p.estimated_parameter_count < 140e6
        )

    def test_moe(self) -> None:
        p = normalize_config("mixtral", load_config("mixtral_8x7b"))
        assert p.is_moe and p.num_experts == 8 and p.num_experts_per_token == 2
        assert p.architecture_summary.startswith("MoE")
        assert p.estimated_parameter_count is not None and 44e9 < p.estimated_parameter_count < 50e9

    def test_deepseek_mla_and_remote_code(self) -> None:
        p = normalize_config("deepseek", load_config("deepseek_v3"))
        assert (
            p.attention_kind == AttentionKind.MLA
            and p.kv_lora_rank == 512
            and p.qk_rope_head_dim == 64
        )
        assert p.is_moe and p.num_experts == 256 and p.num_shared_experts == 1
        assert p.trust_remote_code_required
        assert p.quantization_method == "fp8"

    def test_quantized(self) -> None:
        p = normalize_config("awq", load_config("llama3_70b_awq"))
        assert p.is_quantized and p.quantization_method == "awq"
        assert quantized_bytes_per_parameter(p) == pytest.approx(0.55)

    def test_multimodal_flat_and_nested(self) -> None:
        vl = normalize_config("vl", load_config("qwen2_vl"))
        assert vl.is_multimodal and vl.hidden_size == 3584
        g = normalize_config("gemma", load_config("gemma3_nested"))
        assert g.is_multimodal and g.hidden_size == 5376 and g.num_hidden_layers == 62
        assert g.sliding_window == 1024 and g.configured_dtype == "bfloat16"

    def test_missing_fields(self) -> None:
        p = normalize_config("mystery", load_config("missing_fields"))
        assert p.hidden_size is None and p.estimated_parameter_count is None
        assert p.attention_kind == AttentionKind.UNKNOWN
        assert p.warnings


class TestWeights:
    def test_select_prefers_safetensors_and_ignores_duplicates(self) -> None:
        files, fmt = select_weight_files(
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "pytorch_model.bin",
                "training_args.bin",
                "adapter_model.safetensors",
            ]
        )
        assert fmt == "safetensors" and files == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        files, fmt = select_weight_files(
            ["pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"]
        )
        assert fmt == "torch" and len(files) == 2
        assert select_weight_files(["README.md"]) == ([], "none")

    def test_local_measurement(self, tmp_path: Path) -> None:
        model_dir = make_local_model(
            tmp_path,
            "llama3_8b",
            16 * GIB,
            extra_files={"pytorch_model.bin": 5 * GIB, "optimizer.pt": GIB},
        )
        size = measure_local_weights(model_dir)
        assert size.source == "local_files" and size.total_bytes == 16 * GIB
        profile = inspect_local_model(model_dir)
        assert profile.weight_bytes == 16 * GIB and profile.weight_bytes_is_exact
        assert profile.tokenizer_available and profile.local_path == str(model_dir.resolve())

    def test_local_without_config(self, tmp_path: Path) -> None:
        with pytest.raises(ModelInspectionError):
            inspect_local_model(tmp_path)

    def test_hub_sizes(self) -> None:
        size = measure_hub_weights(
            {
                "model-00001-of-00002.safetensors": 10,
                "model-00002-of-00002.safetensors": 20,
                "pytorch_model.bin": 99,
                "config.json": 1,
            }
        )
        assert size.total_bytes == 30 and size.source == "huggingface_metadata"
        incomplete = measure_hub_weights({"model.safetensors": None})
        assert incomplete.total_bytes is None

    def test_safetensors_counts(self) -> None:
        size = bytes_from_safetensors_parameter_counts({"BF16": 1_000, "F32": 10})
        assert size.total_bytes == 2_040 and size.source == "safetensors_metadata"

    def test_estimate_matches_reality_for_dense(self) -> None:
        p = profile_from_config("llama3_8b", weight_bytes=None)
        est = estimate_weight_bytes(p)
        assert est.source == "estimated"
        assert est.total_bytes is not None and 14.5 * GIB < est.total_bytes < 16.5 * GIB
        p70 = profile_from_config("llama3_70b", weight_bytes=None)
        assert estimate_parameter_count(p70) is not None
        assert 68e9 < (estimate_parameter_count(p70) or 0) < 72e9

    def test_estimate_quantized_uses_bits(self) -> None:
        p = profile_from_config("llama3_70b_awq", weight_bytes=None)
        est = estimate_weight_bytes(p)
        assert est.total_bytes is not None and 33 * GIB < est.total_bytes < 40 * GIB


class _FakeSibling:
    def __init__(self, name: str, size: int | None) -> None:
        self.rfilename = name
        self.size = size


class _FakeApi:
    def __init__(
        self,
        files: dict[str, int | None],
        *,
        sha: str = "abc123",
        error: Exception | None = None,
        param_counts: dict[str, int] | None = None,
    ) -> None:
        self._files = files
        self._sha = sha
        self._error = error
        self._counts = param_counts
        self.calls = 0

    def model_info(
        self, model_id: str, revision: str | None = None, files_metadata: bool = False
    ) -> Any:
        self.calls += 1
        if self._error:
            raise self._error
        return SimpleNamespace(
            sha=self._sha, siblings=[_FakeSibling(n, s) for n, s in self._files.items()]
        )

    def get_safetensors_metadata(self, model_id: str, revision: str | None = None) -> Any:
        if self._counts is None:
            raise RuntimeError("no metadata")
        return SimpleNamespace(parameter_count=self._counts)


class TestHubInspector:
    def _patch_download(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: dict[str, Any]
    ) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(path))

    def test_hub_metadata_sizes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self._patch_download(monkeypatch, tmp_path, load_config("llama3_8b"))
        api = _FakeApi(
            {
                "model-00001-of-00002.safetensors": 8 * GIB,
                "model-00002-of-00002.safetensors": 8 * GIB,
                "tokenizer.json": 1,
            }
        )
        inspector = ModelInspector(hf_api=api)
        p = inspector.inspect("meta-llama/Llama-3-8B")
        assert p.weight_bytes == 16 * GIB and p.weight_size_source == "huggingface_metadata"
        assert p.revision == "abc123" and p.tokenizer_available
        inspector.inspect("meta-llama/Llama-3-8B")
        assert api.calls == 1  # memoised

    def test_hub_falls_back_to_safetensors_metadata_then_estimate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_download(monkeypatch, tmp_path, load_config("llama3_8b"))
        api = _FakeApi({"model.safetensors": None}, param_counts={"BF16": 8_000_000_000})
        p = ModelInspector(hf_api=api).inspect("x/y")
        assert p.weight_size_source == "safetensors_metadata" and p.weight_bytes == 16_000_000_000
        api2 = _FakeApi({"model.safetensors": None})
        p2 = ModelInspector(hf_api=api2).inspect("x/z")
        assert p2.weight_size_source == "estimated" and p2.weight_bytes is not None
        assert any("estimate" in w for w in p2.warnings)

    @staticmethod
    def _hub_error(cls: type[Exception], message: str) -> Exception:
        """Instantiate huggingface_hub errors across versions (newer ones require ``response``)."""
        try:
            return cls(message)
        except TypeError:
            import httpx

            response = httpx.Response(
                403, request=httpx.Request("GET", "https://huggingface.co/api")
            )
            return cls(message, response=response)  # type: ignore[call-arg]

    def test_hub_errors_are_actionable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        self._patch_download(monkeypatch, tmp_path, load_config("llama3_8b"))
        with pytest.raises(ModelInspectionError) as exc:
            ModelInspector(
                hf_api=_FakeApi({}, error=self._hub_error(GatedRepoError, "gated"))
            ).inspect("meta-llama/secret")
        assert "gated" in exc.value.message.lower() and any(
            "HF_TOKEN" in h for h in exc.value.hints
        )
        with pytest.raises(ModelInspectionError) as exc2:
            ModelInspector(
                hf_api=_FakeApi({}, error=self._hub_error(RepositoryNotFoundError, "missing"))
            ).inspect("nope/none")
        assert "not found" in exc2.value.message


class TestFingerprint:
    def test_config_hash_ignores_volatile_keys(self) -> None:
        a = load_config("llama3_8b")
        b = dict(a, transformers_version="9.9.9")
        assert config_hash(a) == config_hash(b)
        c = dict(a, num_hidden_layers=33)
        assert config_hash(a) != config_hash(c)

    def test_revision_changes_fingerprint(self) -> None:
        p1 = profile_from_config("llama3_8b", weight_bytes=16 * GIB)
        p2 = profile_from_config("llama3_8b", weight_bytes=16 * GIB)
        assert model_fingerprint(p1) == model_fingerprint(p2)
        p2.revision = "deadbeef"
        assert model_fingerprint(p1) != model_fingerprint(p2)
