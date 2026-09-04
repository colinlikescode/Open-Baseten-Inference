"""Model inspection from Hugging Face Hub metadata or a local directory.

The inspector never loads weights. It reads ``config.json`` (plus repository file metadata) and
normalises the many architecture-specific key names into a :class:`ModelProfile`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from servepilot.exceptions import ModelInspectionError
from servepilot.logging import get_logger
from servepilot.models.weights import (
    WeightSize,
    bytes_from_safetensors_parameter_counts,
    estimate_parameter_count,
    estimate_weight_bytes,
    measure_hub_weights,
    measure_local_weights,
)
from servepilot.schemas.model import AttentionKind, ModelProfile

log = get_logger(__name__)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "spiece.model",
)

_HIDDEN_KEYS = ("hidden_size", "n_embd", "d_model", "dim")
_INTERMEDIATE_KEYS = ("intermediate_size", "n_inner", "ffn_dim", "ffn_hidden_size", "hidden_dim")
_LAYER_KEYS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")
_HEAD_KEYS = ("num_attention_heads", "n_head", "num_heads", "n_heads")
_KV_HEAD_KEYS = ("num_key_value_heads", "num_kv_heads", "n_head_kv", "num_key_value_groups")
_VOCAB_KEYS = ("vocab_size", "padded_vocab_size")
_MAX_POS_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_seq_len",
    "max_sequence_length",
    "seq_length",
)
_NUM_EXPERTS_KEYS = (
    "num_local_experts",
    "num_experts",
    "n_routed_experts",
    "moe_num_experts",
    "num_moe_experts",
)
_EXPERTS_PER_TOK_KEYS = (
    "num_experts_per_tok",
    "num_experts_per_token",
    "moe_top_k",
    "top_k",
    "n_activated_experts",
)
_SHARED_EXPERTS_KEYS = ("n_shared_experts", "num_shared_experts", "moe_num_shared_experts")
_DTYPE_KEYS = ("torch_dtype", "dtype")
_MULTIMODAL_MARKERS = ("vision_config", "audio_config", "image_token_index", "vision_tower")


def _first(cfg: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def text_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the decoder configuration, descending into ``text_config`` for multimodal wrappers."""
    nested = raw.get("text_config")
    if isinstance(nested, dict) and any(k in nested for k in _HIDDEN_KEYS + _LAYER_KEYS):
        merged = dict(nested)
        # Wrapper-level keys (architectures, torch_dtype, quantization) still apply.
        for key in ("architectures", "quantization_config", "auto_map", *_DTYPE_KEYS):
            if key in raw and key not in merged:
                merged[key] = raw[key]
        return merged
    return raw


def normalize_config(
    model_id: str,
    raw: dict[str, Any],
    *,
    revision: str | None = None,
    local_path: str | None = None,
    tokenizer_available: bool = False,
) -> ModelProfile:
    """Turn a raw ``config.json`` mapping into a :class:`ModelProfile` (without weight sizing)."""
    cfg = text_config(raw)
    architectures = raw.get("architectures") or cfg.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    model_type = raw.get("model_type") or cfg.get("model_type")

    heads = _as_int(_first(cfg, _HEAD_KEYS))
    kv_heads = _as_int(_first(cfg, _KV_HEAD_KEYS))
    if kv_heads is None and cfg.get("multi_query") is True:
        kv_heads = 1
    if kv_heads is None and heads is not None:
        kv_heads = heads
    hidden = _as_int(_first(cfg, _HIDDEN_KEYS))
    head_dim = _as_int(cfg.get("head_dim"))
    if head_dim is None and hidden and heads:
        head_dim = hidden // heads

    num_experts = _as_int(_first(cfg, _NUM_EXPERTS_KEYS))
    experts_per_tok = _as_int(_first(cfg, _EXPERTS_PER_TOK_KEYS))
    is_moe = bool(num_experts and num_experts > 1)

    kv_lora_rank = _as_int(cfg.get("kv_lora_rank"))
    if kv_lora_rank:
        attention = AttentionKind.MLA
    elif heads and kv_heads:
        if kv_heads == heads:
            attention = AttentionKind.MHA
        elif kv_heads == 1:
            attention = AttentionKind.MQA
        else:
            attention = AttentionKind.GQA
    else:
        attention = AttentionKind.UNKNOWN

    dtype = _first(cfg, _DTYPE_KEYS)
    quant = cfg.get("quantization_config") or raw.get("quantization_config")
    if quant is not None and not isinstance(quant, dict):
        quant = {"raw": quant}

    warnings: list[str] = []
    if hidden is None or heads is None or _as_int(_first(cfg, _LAYER_KEYS)) is None:
        warnings.append(
            "configuration is missing core dimensions; memory estimates will be low confidence"
        )
    is_multimodal = any(k in raw for k in _MULTIMODAL_MARKERS) or any(
        "vision" in a.lower() or "forconditionalgeneration" in a.lower() for a in architectures
    )

    tie = cfg.get("tie_word_embeddings")
    profile = ModelProfile(
        model_id=model_id,
        revision=revision,
        local_path=local_path,
        architecture_names=[str(a) for a in architectures],
        model_type=str(model_type) if model_type else None,
        is_moe=is_moe,
        is_multimodal=is_multimodal,
        attention_kind=attention,
        hidden_size=hidden,
        intermediate_size=_as_int(_first(cfg, _INTERMEDIATE_KEYS)),
        num_hidden_layers=_as_int(_first(cfg, _LAYER_KEYS)),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=_as_int(_first(cfg, _VOCAB_KEYS)),
        max_position_embeddings=_as_int(_first(cfg, _MAX_POS_KEYS)),
        sliding_window=_as_int(cfg.get("sliding_window"))
        if cfg.get("use_sliding_window", True)
        else None,
        num_experts=num_experts if is_moe else None,
        num_experts_per_token=experts_per_tok if is_moe else None,
        moe_intermediate_size=_as_int(cfg.get("moe_intermediate_size")) if is_moe else None,
        num_shared_experts=_as_int(_first(cfg, _SHARED_EXPERTS_KEYS)) if is_moe else None,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=_as_int(cfg.get("qk_rope_head_dim")),
        configured_dtype=str(dtype).replace("torch.", "") if dtype else None,
        quantization_config=quant,
        tie_word_embeddings=bool(tie) if tie is not None else None,
        hidden_act=str(cfg.get("hidden_act") or cfg.get("activation_function") or "") or None,
        tokenizer_available=tokenizer_available,
        trust_remote_code_required=bool(raw.get("auto_map") or cfg.get("auto_map")),
        raw_config=raw,
        warnings=warnings,
    )
    profile.estimated_parameter_count = estimate_parameter_count(profile)
    return profile


def apply_weight_size(profile: ModelProfile, size: WeightSize) -> ModelProfile:
    if size.total_bytes is None:
        est = estimate_weight_bytes(profile)
        profile.weight_bytes = est.total_bytes
        profile.weight_size_source = est.source
        if size.note:
            profile.warnings.append(f"exact weight size unavailable ({size.note}); using estimate")
    else:
        profile.weight_bytes = size.total_bytes
        profile.weight_size_source = size.source
    return profile


def inspect_local_model(path: Path, model_id: str | None = None) -> ModelProfile:
    """Inspect a local Hugging Face-compatible directory."""
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ModelInspectionError(
            f"{path} does not contain a config.json",
            hints=[
                "Point ServePilot at the directory that holds config.json and the weight files."
            ],
        )
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelInspectionError(f"could not read {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ModelInspectionError(f"{config_path} does not contain a JSON object")
    tokenizer = any((path / name).is_file() for name in TOKENIZER_FILES)
    profile = normalize_config(
        model_id or str(path),
        raw,
        local_path=str(path.resolve()),
        tokenizer_available=tokenizer,
    )
    return apply_weight_size(profile, measure_local_weights(path))


class ModelInspector:
    """Inspect models by Hub id or local path; results are memoised per instance."""

    def __init__(self, *, token: str | None = None, hf_api: Any | None = None) -> None:
        self._token = token
        self._api = hf_api
        self._cache: dict[tuple[str, str | None], ModelProfile] = {}

    def _hf_api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self._token)
        return self._api

    def inspect(self, model: str, revision: str | None = None) -> ModelProfile:
        key = (model, revision)
        if key in self._cache:
            return self._cache[key]
        path = Path(model).expanduser()
        if path.is_dir():
            profile = inspect_local_model(path, model_id=model)
        elif model.startswith(("/", "./", "../", "~")) or model.count("/") > 1:
            raise ModelInspectionError(
                f"{model!r} looks like a local path but is not a directory containing config.json.",
                hints=[
                    "Check the path, or pass a Hugging Face model id such as Qwen/Qwen3-32B.",
                ],
            )
        else:
            profile = self._inspect_hub(model, revision)
        self._cache[key] = profile
        return profile

    # ------------------------------------------------------------------ hub
    def _inspect_hub(self, model_id: str, revision: str | None) -> ModelProfile:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            HFValidationError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )

        api = self._hf_api()
        try:
            info = api.model_info(model_id, revision=revision, files_metadata=True)
        except HFValidationError as exc:
            raise ModelInspectionError(
                f"{model_id!r} is not a valid Hugging Face model id or local directory.",
                hints=[
                    "Model ids look like `org/name`; local models must be a directory containing config.json."
                ],
            ) from exc
        except GatedRepoError as exc:
            raise ModelInspectionError(
                f"{model_id} is a gated model and your credentials do not grant access.",
                hints=[
                    f"Accept the model's license at https://huggingface.co/{model_id}",
                    "Log in with `huggingface-cli login` or set HF_TOKEN.",
                ],
            ) from exc
        except RepositoryNotFoundError as exc:
            raise ModelInspectionError(
                f"model repository {model_id!r} was not found on the Hugging Face Hub.",
                hints=[
                    "Check the spelling of the model id (it is case sensitive).",
                    "Private repositories require HF_TOKEN with read access.",
                    "For a local model, pass the directory path instead.",
                ],
            ) from exc
        except RevisionNotFoundError as exc:
            raise ModelInspectionError(
                f"revision {revision!r} does not exist for {model_id}.",
                hints=["List branches/tags on the model page and pass a valid --revision."],
            ) from exc
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise ModelInspectionError(
                    f"access to {model_id} was denied (HTTP {status}).",
                    hints=["Set HF_TOKEN to a token with read access to this repository."],
                ) from exc
            raise ModelInspectionError(
                f"Hugging Face Hub request failed for {model_id}: {exc}"
            ) from exc
        except (OSError, LocalEntryNotFoundError) as exc:
            raise ModelInspectionError(
                f"could not reach the Hugging Face Hub to inspect {model_id}: {exc}",
                hints=[
                    "Check network connectivity to https://huggingface.co",
                    "Set HF_HUB_OFFLINE=1 and pass a local model directory to work offline.",
                ],
            ) from exc

        resolved = getattr(info, "sha", None) or revision
        try:
            config_path = hf_hub_download(
                model_id, "config.json", revision=resolved, token=self._token
            )
            raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, HfHubHTTPError) as exc:
            raise ModelInspectionError(
                f"could not download or parse config.json for {model_id}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise ModelInspectionError(f"config.json for {model_id} is not a JSON object")

        siblings = getattr(info, "siblings", None) or []
        sizes: dict[str, int | None] = {}
        for s in siblings:
            name = getattr(s, "rfilename", None)
            if name:
                sizes[str(name)] = getattr(s, "size", None)
        tokenizer = any(name in sizes for name in TOKENIZER_FILES)

        profile = normalize_config(model_id, raw, revision=resolved, tokenizer_available=tokenizer)
        size = measure_hub_weights(sizes)
        if size.total_bytes is None:
            size = self._safetensors_metadata(model_id, resolved) or size
        return apply_weight_size(profile, size)

    def _safetensors_metadata(self, model_id: str, revision: str | None) -> WeightSize | None:
        api = self._hf_api()
        getter = getattr(api, "get_safetensors_metadata", None)
        if getter is None:
            return None
        try:
            meta = getter(model_id, revision=revision)
        except Exception as exc:
            log.debug("safetensors metadata unavailable for %s: %s", model_id, exc)
            return None
        counts = getattr(meta, "parameter_count", None)
        if not isinstance(counts, dict) or not counts:
            return None
        return bytes_from_safetensors_parameter_counts({str(k): int(v) for k, v in counts.items()})
