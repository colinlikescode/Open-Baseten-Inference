"""Hardware provider protocol and factory.

The planner, tuner and runtime only ever talk to :class:`HardwareProvider`. Production uses
:class:`servepilot.hardware.nvml.NVMLHardwareProvider`; tests and the CLI's testing mode use
:class:`servepilot.testing.fake_hardware.FakeHardwareProvider`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from servepilot.schemas.hardware import GPUSample, HardwareSnapshot

if TYPE_CHECKING:
    from servepilot.settings import ServePilotSettings


@runtime_checkable
class HardwareProvider(Protocol):
    """Source of hardware snapshots and utilization samples."""

    @property
    def name(self) -> str: ...

    def snapshot(self) -> HardwareSnapshot:
        """Return a fresh snapshot (free memory is re-read every call)."""
        ...

    def sample(self, gpu_indices: Sequence[int]) -> list[GPUSample]:
        """Return utilization samples for the given GPUs. May return an empty list."""
        ...


def parse_cuda_visible_devices(value: str | None) -> list[str] | None:
    """Parse ``CUDA_VISIBLE_DEVICES``; returns None when unset/empty-meaning-all."""
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items


def apply_cuda_visible_devices(
    snapshot: HardwareSnapshot, visible: list[str] | None
) -> HardwareSnapshot:
    """Restrict a snapshot to the devices named in ``CUDA_VISIBLE_DEVICES``.

    Entries may be NVML indices or ``GPU-<uuid>`` strings. Unknown entries are ignored, which
    mirrors CUDA's behaviour of silently dropping invalid devices.
    """
    if visible is None:
        return snapshot
    keep: list[int] = []
    for item in visible:
        if item.isdigit():
            idx = int(item)
            if any(g.index == idx for g in snapshot.gpus):
                keep.append(idx)
        else:
            for g in snapshot.gpus:
                if g.uuid.lower() == item.lower():
                    keep.append(g.index)
    return snapshot.select(keep)


def get_hardware_provider(settings: ServePilotSettings | None = None) -> HardwareProvider:
    """Return the configured provider (fake when ``SERVEPILOT_FAKE_HARDWARE`` is set)."""
    from servepilot.settings import ServePilotSettings

    settings = settings or ServePilotSettings()
    if settings.fake_hardware:
        from servepilot.testing.fake_hardware import FakeHardwareProvider, fixture_by_name

        return FakeHardwareProvider(fixture_by_name(settings.fake_hardware))
    from servepilot.hardware.nvml import NVMLHardwareProvider

    return NVMLHardwareProvider(
        cuda_visible_devices=parse_cuda_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    )
