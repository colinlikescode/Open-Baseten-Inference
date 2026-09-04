"""ServePilot: automatically determine the fastest way to serve an LLM on your NVIDIA GPUs."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("servepilot")
except PackageNotFoundError:  # pragma: no cover - only when running from an unbuilt checkout
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
