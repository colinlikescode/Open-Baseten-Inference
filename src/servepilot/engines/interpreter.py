"""Locate the Python interpreter that has an engine installed.

Inference stacks (vLLM, SGLang) often live in dedicated virtual environments because their CUDA
dependencies conflict. ServePilot therefore supports running an engine through *any* interpreter:

1. ``SERVEPILOT_<ENGINE>_PYTHON`` – explicit interpreter path.
2. The current interpreter, when the engine's distribution is importable.
3. A console script on ``PATH`` (e.g. ``vllm``) – its shebang reveals the interpreter.
4. Conventional locations such as ``~/engines/<engine>/bin/python``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from servepilot.logging import get_logger

log = get_logger(__name__)

_VERSION_SNIPPET = "import importlib.metadata as m, sys; sys.stdout.write(m.version(sys.argv[1]))"


@dataclass(frozen=True)
class EngineRuntime:
    python: str
    version: str


def _shebang_interpreter(script: Path) -> str | None:
    try:
        with script.open("rb") as fh:
            first = fh.readline(512)
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    tokens = first[2:].decode("utf-8", errors="replace").split()
    if not tokens:
        return None
    candidate = tokens[0]
    if Path(candidate).name == "env" and len(tokens) > 1:
        # ``#!/usr/bin/env python3``: the interpreter is whatever PATH resolves next.
        return shutil.which(tokens[1])
    return candidate if Path(candidate).exists() else None


@lru_cache(maxsize=32)
def _external_version(python: str, distribution: str) -> str | None:
    try:
        proc = subprocess.run(
            [python, "-c", _VERSION_SNIPPET, distribution],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.debug("version probe failed for %s via %s: %s", distribution, python, exc)
        return None
    if proc.returncode != 0:
        return None
    version = proc.stdout.strip()
    return version or None


def find_engine_runtime(
    distribution: str,
    *,
    env_var: str,
    console_scripts: tuple[str, ...] = (),
    conventional_dirs: tuple[str, ...] = (),
) -> EngineRuntime | None:
    """Return the interpreter + version for ``distribution`` or None when not installed."""
    explicit = os.environ.get(env_var)
    if explicit:
        version = _external_version(explicit, distribution)
        if version:
            return EngineRuntime(explicit, version)
        log.warning("%s=%s does not provide %s; ignoring", env_var, explicit, distribution)
        return None

    try:
        return EngineRuntime(sys.executable, metadata.version(distribution))
    except metadata.PackageNotFoundError:
        pass

    for script in console_scripts:
        found = shutil.which(script)
        if found:
            interp = _shebang_interpreter(Path(found))
            if interp:
                version = _external_version(interp, distribution)
                if version:
                    return EngineRuntime(interp, version)

    for directory in conventional_dirs:
        python = Path(directory).expanduser() / "bin" / "python"
        if python.exists():
            version = _external_version(str(python), distribution)
            if version:
                return EngineRuntime(str(python), version)
    return None


def parse_version(version: str | None) -> tuple[int, ...]:
    """Parse ``0.28.0.post1`` → ``(0, 28, 0)``; unparseable → ``()``."""
    if not version:
        return ()
    parts: list[int] = []
    for chunk in version.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)
