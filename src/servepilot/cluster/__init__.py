"""Ray cluster support: cluster-wide hardware discovery and remote engine process placement.

Ray is an optional dependency (``pip install "servepilot[ray]"``). Everything in this package
imports Ray lazily so the rest of ServePilot works without it.
"""

from __future__ import annotations

from typing import Any

from servepilot.exceptions import EngineUnavailableError


def require_ray() -> Any:
    """Import and return the ``ray`` module, or raise an actionable error."""
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise EngineUnavailableError(
            "Ray is not installed but a Ray cluster address was given.",
            hints=[
                'pip install "servepilot[ray]"',
                "Ray must also be installed on every worker node.",
            ],
        ) from exc
    return ray
