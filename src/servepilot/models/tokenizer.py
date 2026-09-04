"""Tokenizer access for benchmark prompt generation and token accounting.

Preference order: ``tokenizer.json`` through the fast ``tokenizers`` library (works for nearly
all modern checkpoints), then ``transformers.AutoTokenizer`` when installed (sentencepiece-only
checkpoints), then a deterministic approximation that is clearly labelled as such.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Protocol

from servepilot.logging import get_logger
from servepilot.schemas.model import ModelProfile

log = get_logger(__name__)


class TokenCounter(Protocol):
    name: str
    exact: bool

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def count(self, text: str) -> int: ...


class FastTokenizer:
    """Wraps a ``tokenizers.Tokenizer`` loaded from ``tokenizer.json``."""

    exact = True

    def __init__(self, tokenizer: Any, name: str) -> None:
        self._tok = tokenizer
        self.name = name

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return str(self._tok.decode(ids, skip_special_tokens=True))

    def count(self, text: str) -> int:
        return len(self.encode(text))


class TransformersTokenizer:
    """Wraps a ``transformers`` tokenizer (slow or fast)."""

    exact = True

    def __init__(self, tokenizer: Any, name: str) -> None:
        self._tok = tokenizer
        self.name = name

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return str(self._tok.decode(ids, skip_special_tokens=True))

    def count(self, text: str) -> int:
        return len(self.encode(text))


class ApproximateTokenizer:
    """Deterministic fallback: ~1.3 tokens per word, punctuation counted separately.

    Used only when no real tokenizer can be loaded; every consumer surfaces ``exact=False`` so
    token-derived metrics are reported as approximate.
    """

    exact = False
    name = "approximate"
    _word_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in self._word_re.finditer(text):
            token = match.group(0)
            pieces = max(1, math.ceil(len(token) / 4)) if token.isalnum() else 1
            ids.extend(hash(token) % 50_000 + 1 for _ in range(pieces))
        return ids

    def decode(self, ids: list[int]) -> str:  # pragma: no cover - never used for text generation
        return " ".join("tok" for _ in ids)

    def count(self, text: str) -> int:
        return len(self.encode(text))


def _load_fast(path: Path, name: str) -> TokenCounter | None:
    try:
        from tokenizers import Tokenizer
    except ImportError:  # pragma: no cover - declared dependency
        return None
    try:
        return FastTokenizer(Tokenizer.from_file(str(path)), name)
    except Exception as exc:
        log.debug("tokenizers could not load %s: %s", path, exc)
        return None


def _load_transformers(
    source: str, revision: str | None, trust_remote_code: bool, token: str | None
) -> TokenCounter | None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(
            source, revision=revision, trust_remote_code=trust_remote_code, token=token
        )
        return TransformersTokenizer(tok, f"transformers:{source}")
    except Exception as exc:
        log.debug("transformers could not load tokenizer %s: %s", source, exc)
        return None


def load_tokenizer(
    model: ModelProfile,
    *,
    token: str | None = None,
    trust_remote_code: bool = False,
    allow_download: bool = True,
) -> TokenCounter:
    """Load the best available tokenizer for ``model``; never raises."""
    source = model.local_path or model.model_id
    if model.local_path:
        local = Path(model.local_path) / "tokenizer.json"
        if local.is_file():
            tok = _load_fast(local, f"tokenizer.json:{source}")
            if tok is not None:
                return tok
    elif allow_download:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                model.model_id, "tokenizer.json", revision=model.revision, token=token
            )
            tok = _load_fast(Path(path), f"tokenizer.json:{source}")
            if tok is not None:
                return tok
        except Exception as exc:
            log.debug("tokenizer.json unavailable for %s: %s", source, exc)

    tok = _load_transformers(source, model.revision, trust_remote_code, token)
    if tok is not None:
        return tok

    log.warning(
        "No tokenizer could be loaded for %s; benchmark token counts will be approximate "
        "(install `servepilot[hf]` for full tokenizer coverage).",
        source,
    )
    return ApproximateTokenizer()
