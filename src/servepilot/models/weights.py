"""Determine how many bytes of weights a model has.

Priority (see spec §10):

1. ``local_files``            – sum the actual weight files of a local checkout.
2. ``huggingface_metadata``   – byte sizes reported by the Hub for the repository's weight files.
3. ``safetensors_metadata``   – parameter counts per dtype from safetensors headers.
4. ``estimated``              – derived from the architecture configuration.

Every result carries its provenance so an estimate is never presented as an exact number.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from servepilot.schemas.model import ModelProfile, WeightSizeSource, dtype_bytes

SAFETENSORS_SUFFIXES = (".safetensors",)
TORCH_SUFFIXES = (".bin", ".pt", ".pth")
GGUF_SUFFIXES = (".gguf",)
# Files that are weights but should never be counted as *model* weights.
EXCLUDED_NAME_PATTERNS = (
    re.compile(r"^training_args"),
    re.compile(r"optimizer"),
    re.compile(r"scheduler"),
    re.compile(r"rng_state"),
    re.compile(r"^adapter_"),  # LoRA adapters live in their own repos; ignore if present
    re.compile(r"consolidated\.\d+\.pth$"),  # duplicate raw-torch shards shipped alongside HF ones
)


@dataclass(frozen=True)
class WeightSize:
    total_bytes: int | None
    source: WeightSizeSource
    files: tuple[str, ...] = ()
    note: str = ""


def _is_excluded(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return any(p.search(base) for p in EXCLUDED_NAME_PATTERNS)


def classify_weight_files(names: Iterable[str]) -> dict[str, list[str]]:
    """Bucket file names into weight formats (``safetensors``, ``torch``, ``gguf``)."""
    buckets: dict[str, list[str]] = {"safetensors": [], "torch": [], "gguf": []}
    for name in names:
        if _is_excluded(name):
            continue
        lower = name.lower()
        if lower.endswith(SAFETENSORS_SUFFIXES):
            buckets["safetensors"].append(name)
        elif lower.endswith(TORCH_SUFFIXES):
            buckets["torch"].append(name)
        elif lower.endswith(GGUF_SUFFIXES):
            buckets["gguf"].append(name)
    return buckets


def select_weight_files(names: Iterable[str]) -> tuple[list[str], str]:
    """Choose one weight representation so duplicates (``.bin`` and ``.safetensors``) don't double count.

    Returns ``(files, format)``; safetensors is preferred, then torch, then GGUF.
    """
    buckets = classify_weight_files(names)
    for fmt in ("safetensors", "torch", "gguf"):
        if buckets[fmt]:
            return sorted(buckets[fmt]), fmt
    return [], "none"


def measure_local_weights(path: Path) -> WeightSize:
    """Sum the bytes of the weight files in a local model directory (Level 1)."""
    if not path.is_dir():
        return WeightSize(None, "unknown", note=f"{path} is not a directory")
    names = [p.name for p in path.iterdir() if p.is_file()]
    files, fmt = select_weight_files(names)
    if not files:
        return WeightSize(None, "unknown", note="no weight files found")
    total = sum((path / f).stat().st_size for f in files)
    return WeightSize(total, "local_files", tuple(files), note=f"{len(files)} {fmt} file(s)")


def measure_hub_weights(sizes: Mapping[str, int | None]) -> WeightSize:
    """Sum Hub-reported file sizes for the selected weight representation (Level 2)."""
    files, fmt = select_weight_files(sizes.keys())
    if not files:
        return WeightSize(None, "unknown", note="repository lists no weight files")
    missing = [f for f in files if sizes.get(f) is None]
    if missing:
        return WeightSize(
            None,
            "unknown",
            tuple(files),
            note=f"Hub metadata lacks sizes for {len(missing)} of {len(files)} files",
        )
    total = sum(int(sizes[f] or 0) for f in files)
    return WeightSize(
        total, "huggingface_metadata", tuple(files), note=f"{len(files)} {fmt} file(s)"
    )


def bytes_from_safetensors_parameter_counts(counts: Mapping[str, int]) -> WeightSize:
    """Convert a ``{dtype: parameter_count}`` map (safetensors metadata) into bytes (Level 2b)."""
    total = 0.0
    for dtype, count in counts.items():
        per = dtype_bytes(dtype)
        if per is None:
            # Unknown dtype in the header: fall back to 2 bytes but flag reduced confidence.
            per = 2.0
        total += count * per
    if total <= 0:
        return WeightSize(None, "unknown", note="safetensors metadata reported no parameters")
    return WeightSize(
        int(total), "safetensors_metadata", note=f"{sum(counts.values()):,} parameters"
    )


_NON_GATED_ACTIVATIONS = {
    "gelu",
    "gelu_new",
    "gelu_fast",
    "gelu_pytorch_tanh",
    "quick_gelu",
    "relu",
}


def estimate_parameter_count(profile: ModelProfile) -> int | None:
    """Estimate the parameter count of a decoder-only transformer from its configuration (Level 3).

    Embeddings + (optional untied) LM head + per-layer attention + MLP/experts + norms. Gated
    MLPs (SwiGLU) use three projections; classic GELU/ReLU MLPs use two.
    """
    h = profile.hidden_size
    layers = profile.num_hidden_layers
    heads = profile.num_attention_heads
    if not h or not layers or not heads:
        return None
    kv_heads = profile.num_key_value_heads or heads
    head_dim = profile.head_dim or (h // heads)
    vocab = profile.vocab_size or 0

    embeddings = vocab * h
    lm_head = 0 if profile.tie_word_embeddings else vocab * h

    attn = h * heads * head_dim + 2 * h * kv_heads * head_dim + heads * head_dim * h
    if profile.attention_kind.value == "mla" and profile.kv_lora_rank:
        # DeepSeek-style MLA: q/kv down-projections + up-projections. Approximate with the
        # standard attention size which is a close upper bound for planning purposes.
        attn = int(attn * 0.9)

    gated = (profile.hidden_act or "silu").lower() not in _NON_GATED_ACTIVATIONS
    mats = 3 if gated else 2
    if profile.is_moe and profile.num_experts:
        inter = profile.moe_intermediate_size or profile.intermediate_size or 0
        experts = profile.num_experts * mats * h * inter
        shared = (profile.num_shared_experts or 0) * mats * h * inter
        router = h * profile.num_experts
        mlp = experts + shared + router
    else:
        inter = profile.intermediate_size or 4 * h
        mlp = mats * h * inter
    norms = 2 * h  # pre-attention and pre-MLP RMSNorm/LayerNorm weights
    per_layer = attn + mlp + norms
    return int(embeddings + lm_head + layers * per_layer + h)


def estimate_weight_bytes(profile: ModelProfile) -> WeightSize:
    """Level 3: parameter count × bytes per parameter (quantization aware)."""
    params = estimate_parameter_count(profile)
    if params is None:
        return WeightSize(None, "unknown", note="configuration lacks hidden_size/layers/heads")
    per_param = quantized_bytes_per_parameter(profile)
    if per_param is None:
        per_param = dtype_bytes(profile.configured_dtype) or 2.0
    return WeightSize(
        int(params * per_param),
        "estimated",
        note=f"~{params / 1e9:.1f}B parameters × {per_param} B",
    )


def quantized_bytes_per_parameter(profile: ModelProfile) -> float | None:
    """Bytes per parameter implied by ``quantization_config`` (None when not quantized/unknown)."""
    q = profile.quantization_config
    if not q:
        return None
    for key in ("bits", "weight_bits", "w_bit", "num_bits"):
        value = q.get(key)
        if isinstance(value, int | float) and value > 0:
            # Add ~10% for scales/zero points, which are stored alongside packed weights.
            return float(value) / 8.0 * 1.1
    method = str(q.get("quant_method", "")).lower()
    if "fp8" in method or "float8" in method:
        return 1.0 * 1.02
    if method in {"awq", "gptq", "compressed-tensors", "compressed_tensors"}:
        config_groups = q.get("config_groups")
        if isinstance(config_groups, dict):
            for group in config_groups.values():
                weights = group.get("weights") if isinstance(group, dict) else None
                if isinstance(weights, dict) and isinstance(weights.get("num_bits"), int):
                    return float(weights["num_bits"]) / 8.0 * 1.1
        if method in {"awq", "gptq"}:
            return 0.5 * 1.1
    if method in {"bitsandbytes", "bnb"}:
        if q.get("load_in_4bit"):
            return 0.5 * 1.1
        if q.get("load_in_8bit"):
            return 1.0 * 1.05
    return None
