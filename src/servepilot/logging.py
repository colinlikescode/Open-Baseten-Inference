"""Structured logging configuration.

Human-readable output goes through Rich on stderr; ``--log-format json`` emits one JSON object
per line. All handlers pass through :class:`SecretRedactionFilter` so access tokens never reach
a terminal or log file.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any, Literal

from rich.console import Console
from rich.logging import RichHandler

LogFormat = Literal["human", "json"]

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Hugging Face tokens (user/org/read/write tokens all start with hf_).
    re.compile(r"hf_[A-Za-z0-9]{8,}"),
    # KEY=value forms for well-known secret environment variables.
    re.compile(
        r"((?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACEHUB_API_TOKEN|OPENAI_API_KEY|"
        r"SERVEPILOT_API_KEY)\s*[=:]\s*)(\S+)"
    ),
    # Authorization headers.
    re.compile(r"(Bearer\s+)([A-Za-z0-9._\-]{8,})"),
)

REDACTED = "***REDACTED***"


def redact_secrets(text: str) -> str:
    """Replace any recognizable secret in ``text`` with a redaction marker."""
    result = _SECRET_PATTERNS[0].sub(REDACTED, text)
    result = _SECRET_PATTERNS[1].sub(lambda m: f"{m.group(1)}{REDACTED}", result)
    result = _SECRET_PATTERNS[2].sub(lambda m: f"{m.group(1)}{REDACTED}", result)
    return result


class SecretRedactionFilter(logging.Filter):
    """Logging filter that redacts secrets from the formatted message and arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    redact_secrets(a) if isinstance(a, str) else a for a in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: (redact_secrets(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
        return True


class JSONFormatter(logging.Formatter):
    """Format records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("candidate_id", "pid", "port", "replica_id", "event"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info and record.exc_info[1] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def verbosity_to_level(verbosity: int) -> int:
    """Map ``-v`` counts to logging levels: 0 → WARNING, 1 → INFO, 2+ → DEBUG."""
    if verbosity <= 0:
        return logging.WARNING
    if verbosity == 1:
        return logging.INFO
    return logging.DEBUG


def configure_logging(verbosity: int = 0, log_format: LogFormat = "human") -> None:
    """Configure the root ``servepilot`` logger.

    Logging always goes to stderr so ``--json`` command output on stdout stays machine-readable.
    """
    level = verbosity_to_level(verbosity)
    root = logging.getLogger("servepilot")
    root.setLevel(level)
    for existing in list(root.handlers):
        root.removeHandler(existing)

    handler: logging.Handler
    if log_format == "json":
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(JSONFormatter())
        handler = stream_handler
    else:
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_time=verbosity >= 2,
            show_path=verbosity >= 2,
            rich_tracebacks=verbosity >= 2,
            markup=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(SecretRedactionFilter())
    handler.setLevel(level)
    root.addHandler(handler)
    root.propagate = False

    # Quiet noisy third-party loggers unless the user asked for debug output.
    for noisy in ("httpx", "httpcore", "uvicorn", "uvicorn.access", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if verbosity >= 3 else logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger in the ``servepilot`` namespace."""
    if name.startswith("servepilot"):
        return logging.getLogger(name)
    return logging.getLogger(f"servepilot.{name}")
