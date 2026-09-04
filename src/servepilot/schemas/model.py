"""Model profile schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

WeightSizeSource = Literal[
    "local_files",
    "huggingface_metadata",
    "safetensors_metadata",
    "estimated",
    "unknown",
]


class AttentionKind(StrEnum):
    MHA = "mha"
    GQA = "gqa"
    MQA = "mqa"
    MLA = "mla"
    UNKNOWN = "unknown"


class ModelProfile(BaseModel):
    """Normalized description of an LLM derived from its configuration, never its weights."""

    model_id: str
    revision: str | None = None
    local_path: str | None = None

    architecture_names: list[str] = Field(default_factory=list)
    model_type: str | None = None

    is_moe: bool = False
    is_multimodal: bool = False
    attention_kind: AttentionKind = AttentionKind.UNKNOWN

    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None

    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    sliding_window: int | None = None

    num_experts: int | None = None
    num_experts_per_token: int | None = None
    moe_intermediate_size: int | None = None
    num_shared_experts: int | None = None

    # MLA (DeepSeek-style) specific dimensions.
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None

    configured_dtype: str | None = None
    quantization_config: dict[str, Any] | None = None
    tie_word_embeddings: bool | None = None
    hidden_act: str | None = None

    weight_bytes: int | None = None
    weight_size_source: WeightSizeSource = "unknown"
    estimated_parameter_count: int | None = None

    tokenizer_available: bool = False
    trust_remote_code_required: bool = False

    raw_config: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.model_id

    @property
    def is_quantized(self) -> bool:
        return bool(self.quantization_config)

    @property
    def quantization_method(self) -> str | None:
        if not self.quantization_config:
            return None
        method = self.quantization_config.get("quant_method")
        return str(method) if method is not None else None

    @property
    def weight_bytes_is_exact(self) -> bool:
        return self.weight_size_source in {"local_files", "huggingface_metadata"}

    @property
    def effective_kv_heads(self) -> int | None:
        return self.num_key_value_heads or self.num_attention_heads

    @property
    def architecture_summary(self) -> str:
        if self.is_moe:
            experts = f"{self.num_experts} experts" if self.num_experts else "experts unknown"
            active = (
                f", {self.num_experts_per_token} active/token" if self.num_experts_per_token else ""
            )
            return f"MoE decoder ({experts}{active})"
        if self.is_multimodal:
            return "multimodal decoder"
        return "dense decoder"


def dtype_bytes(dtype: str | None) -> float | None:
    """Return the bytes per element for a dtype string, or None when unknown."""
    if dtype is None:
        return None
    d = dtype.lower().replace("torch.", "")
    table: dict[str, float] = {
        # PyTorch / config.json names
        "float64": 8,
        "float32": 4,
        "fp32": 4,
        "float16": 2,
        "fp16": 2,
        "half": 2,
        "bfloat16": 2,
        "bf16": 2,
        "float8": 1,
        "fp8": 1,
        "float8_e4m3fn": 1,
        "float8_e5m2": 1,
        "fp8_e4m3": 1,
        "fp8_e5m2": 1,
        "int8": 1,
        "uint8": 1,
        "int32": 4,
        "int64": 8,
        "int4": 0.5,
        "uint4": 0.5,
        "nf4": 0.5,
        "fp4": 0.5,
        # safetensors header names
        "f64": 8,
        "f32": 4,
        "f16": 2,
        "f8_e4m3": 1,
        "f8_e5m2": 1,
        "i8": 1,
        "u8": 1,
        "i32": 4,
        "i64": 8,
        "bool": 1,
    }
    return table.get(d)
