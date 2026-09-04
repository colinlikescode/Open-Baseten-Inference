"""Stable hardware identity used to key cached tuning results.

Volatile facts (free memory, temperatures, capture time, hostname) are deliberately excluded so
that a plan tuned yesterday still matches today. Free memory is validated separately before a
cached plan is reused.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from servepilot.schemas.hardware import HardwareSnapshot


def hardware_identity(
    snapshot: HardwareSnapshot, gpu_ids: Sequence[int] | None = None
) -> dict[str, object]:
    """Return the JSON-serialisable identity dictionary that is hashed into the fingerprint."""
    selected = snapshot.select(list(gpu_ids)) if gpu_ids is not None else snapshot
    gpus = sorted(selected.gpus, key=lambda g: g.index)
    edges = sorted(
        (
            {
                "a": e.gpu_a,
                "b": e.gpu_b,
                "rel": e.relationship,
                "nvlink": e.nvlink_detected,
                "links": e.nvlink_link_count,
            }
            for e in selected.topology.edges
        ),
        key=lambda e: (int(e["a"]), int(e["b"])),  # type: ignore[arg-type]
    )
    return {
        "platform": snapshot.platform,
        "driver_version": snapshot.driver_version,
        "cuda_version": snapshot.cuda_version,
        "gpus": [
            {
                "index": g.index,
                "name": g.name,
                "total_memory_bytes": g.total_memory_bytes,
                "compute_capability": g.compute_capability,
                "mig_mode": g.mig_mode,
            }
            for g in gpus
        ],
        "topology_available": selected.topology.available,
        "topology": edges,
    }


def hardware_fingerprint(snapshot: HardwareSnapshot, gpu_ids: Sequence[int] | None = None) -> str:
    """SHA-256 hex digest of :func:`hardware_identity`."""
    payload = json.dumps(
        hardware_identity(snapshot, gpu_ids), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
