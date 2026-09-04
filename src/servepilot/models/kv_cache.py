"""KV-cache size estimators.

Each estimator returns bytes per cached token for the whole model and per GPU at a given tensor
parallel size, plus a confidence level. Unknown architectures return ``None`` sizes so the planner
relies on launch validation instead of a fabricated number.
"""

from __future__ import annotations

from typing import Protocol

from servepilot.schemas.model import AttentionKind, ModelProfile, dtype_bytes
from servepilot.schemas.plan import KVEstimate


def kv_element_bytes(kv_cache_dtype: str | None, model_dtype: str | None) -> float:
    """Bytes per KV element: explicit KV dtype (e.g. fp8) wins over the model dtype."""
    if kv_cache_dtype and kv_cache_dtype.lower() != "auto":
        per = dtype_bytes(kv_cache_dtype)
        if per is not None:
            return per
    return dtype_bytes(model_dtype) or 2.0


class KVCacheEstimator(Protocol):
    name: str

    def estimate(
        self, model: ModelProfile, tp_size: int, kv_cache_dtype: str | None
    ) -> KVEstimate: ...


class StandardAttentionKVEstimator:
    """MHA / GQA / MQA: ``2 × layers × kv_heads × head_dim × bytes`` per token.

    Per GPU, engines shard KV heads across tensor-parallel ranks; when ``kv_heads < tp`` the heads
    are replicated, so each rank holds ``max(kv_heads / tp, 1)`` heads.
    """

    name = "standard_attention"

    def estimate(self, model: ModelProfile, tp_size: int, kv_cache_dtype: str | None) -> KVEstimate:
        layers = model.num_hidden_layers
        kv_heads = model.effective_kv_heads
        heads = model.num_attention_heads
        head_dim = model.head_dim or (
            model.hidden_size // heads if model.hidden_size and heads else None
        )
        if not layers or not kv_heads or not head_dim:
            return KVEstimate(
                confidence="unknown", explanation="configuration lacks layers/kv heads/head_dim"
            )
        per_elem = kv_element_bytes(kv_cache_dtype, model.configured_dtype)
        total = int(2 * layers * kv_heads * head_dim * per_elem)
        heads_per_rank = max(kv_heads / tp_size, 1.0)
        per_gpu = int(2 * layers * heads_per_rank * head_dim * per_elem)
        confidence: str = "high"
        notes = [
            f"2 × {layers} layers × {kv_heads} KV heads × {head_dim} head_dim × {per_elem:g} B = {total:,} B/token"
        ]
        if model.head_dim is None:
            confidence = "medium"
            notes.append("head_dim inferred from hidden_size / num_attention_heads")
        if model.sliding_window:
            confidence = "medium"
            notes.append(
                f"sliding window {model.sliding_window} may reduce actual KV usage; estimate is an upper bound"
            )
        if kv_heads < tp_size:
            notes.append(f"KV heads replicated across TP ranks ({kv_heads} heads < TP {tp_size})")
        return KVEstimate(
            bytes_per_token_total=total,
            bytes_per_token_per_gpu=per_gpu,
            confidence=confidence,  # type: ignore[arg-type]
            explanation="; ".join(notes),
        )


class MLAKVEstimator:
    """DeepSeek-style Multi-head Latent Attention.

    Engines cache the compressed latent (``kv_lora_rank``) plus the decoupled RoPE key
    (``qk_rope_head_dim``) per layer, and replicate it across TP ranks. Reported with medium
    confidence because engine-specific layouts differ.
    """

    name = "mla"

    def estimate(self, model: ModelProfile, tp_size: int, kv_cache_dtype: str | None) -> KVEstimate:
        layers = model.num_hidden_layers
        if not layers or not model.kv_lora_rank:
            return KVEstimate(
                confidence="unknown", explanation="MLA configuration lacks kv_lora_rank/layers"
            )
        per_elem = kv_element_bytes(kv_cache_dtype, model.configured_dtype)
        latent = model.kv_lora_rank + (model.qk_rope_head_dim or 0)
        total = int(layers * latent * per_elem)
        return KVEstimate(
            bytes_per_token_total=total,
            bytes_per_token_per_gpu=total,  # latent KV is replicated on every TP rank
            confidence="medium",
            explanation=(
                f"MLA: {layers} layers × ({model.kv_lora_rank} latent + {model.qk_rope_head_dim or 0} rope) "
                f"× {per_elem:g} B = {total:,} B/token, replicated per TP rank"
            ),
        )


class UnknownKVEstimator:
    """Fallback for architectures we cannot reason about (hybrid SSM, recurrent, ...)."""

    name = "unknown"

    def estimate(self, model: ModelProfile, tp_size: int, kv_cache_dtype: str | None) -> KVEstimate:
        return KVEstimate(
            confidence="unknown",
            explanation=(
                f"no KV-cache formula for architecture {model.model_type or model.architecture_names or 'unknown'}; "
                "capacity will be established by launch validation"
            ),
        )


# Architectures with non-standard cache layouts (state-space / hybrid / recurrent).
NON_STANDARD_MODEL_TYPES = {
    "mamba",
    "mamba2",
    "jamba",
    "falcon_mamba",
    "recurrent_gemma",
    "rwkv",
    "rwkv6",
    "zamba",
    "zamba2",
    "bamba",
    "nemotron_h",
    "granitemoehybrid",
}


def select_kv_estimator(model: ModelProfile) -> KVCacheEstimator:
    """Pick the estimator appropriate for ``model``."""
    if (model.model_type or "").lower() in NON_STANDARD_MODEL_TYPES:
        return UnknownKVEstimator()
    if model.attention_kind == AttentionKind.MLA:
        return MLAKVEstimator()
    if (
        model.num_hidden_layers
        and model.effective_kv_heads
        and (model.head_dim or model.hidden_size)
    ):
        return StandardAttentionKVEstimator()
    return UnknownKVEstimator()


def estimate_kv(model: ModelProfile, tp_size: int, kv_cache_dtype: str | None = None) -> KVEstimate:
    return select_kv_estimator(model).estimate(model, tp_size, kv_cache_dtype)
