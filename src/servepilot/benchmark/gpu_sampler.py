"""Background GPU utilization sampling during benchmarks."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence

from servepilot.constants import GPU_SAMPLE_INTERVAL_SECONDS
from servepilot.hardware.base import HardwareProvider
from servepilot.logging import get_logger
from servepilot.schemas.hardware import GPUSample

log = get_logger(__name__)


class GPUSampler:
    """Samples ``provider.sample()`` on an interval; failures never abort a benchmark."""

    def __init__(
        self,
        provider: HardwareProvider | None,
        gpu_indices: Sequence[int],
        interval_seconds: float = GPU_SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        self._provider = provider
        self._indices = list(gpu_indices)
        self._interval = interval_seconds
        self.samples: list[GPUSample] = []
        self._task: asyncio.Task[None] | None = None
        self._failed = False

    async def _loop(self) -> None:
        while True:
            try:
                batch = await asyncio.to_thread(self._provider.sample, self._indices)  # type: ignore[union-attr]
                self.samples.extend(batch)
            except Exception as exc:
                if not self._failed:
                    log.debug("GPU sampling failed; continuing without metrics: %s", exc)
                self._failed = True
            await asyncio.sleep(self._interval)

    async def __aenter__(self) -> GPUSampler:
        if self._provider is not None and self._indices:
            self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
