"""Stable model identity for the tuning cache."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from servepilot.schemas.model import ModelProfile

# Keys that change between transformers releases without changing the model.
_VOLATILE_CONFIG_KEYS = {"transformers_version", "_name_or_path", "_commit_hash", "use_cache"}


def config_hash(raw_config: dict[str, Any]) -> str:
    cleaned = {k: v for k, v in raw_config.items() if k not in _VOLATILE_CONFIG_KEYS}
    payload = json.dumps(cleaned, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_identity(profile: ModelProfile) -> dict[str, Any]:
    return {
        "model_id": profile.model_id,
        "revision": profile.revision,
        "config_hash": config_hash(profile.raw_config),
        "weight_bytes": profile.weight_bytes,
        "weight_size_source": profile.weight_size_source,
        "quantization": profile.quantization_config,
        "dtype": profile.configured_dtype,
    }


def model_fingerprint(profile: ModelProfile) -> str:
    payload = json.dumps(
        model_identity(profile), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
