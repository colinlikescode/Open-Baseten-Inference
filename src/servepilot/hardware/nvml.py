"""NVML-backed hardware provider.

NVML is initialised lazily once per provider and shut down at interpreter exit. Every
per-device query that may legitimately be unsupported (MIG, NUMA, NVLink, power) is isolated so
that a partial failure degrades a field to ``None`` instead of making ServePilot unusable.
"""

from __future__ import annotations

import atexit
import contextlib
import platform
import socket
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from servepilot.exceptions import HardwareError
from servepilot.hardware.base import apply_cuda_visible_devices
from servepilot.hardware.topology import (
    ancestor_to_relationship,
    build_topology,
    unavailable_topology,
)
from servepilot.logging import get_logger
from servepilot.schemas.hardware import GPUDevice, GPUSample, GPUTopology, HardwareSnapshot

log = get_logger(__name__)

NVML_INSTALL_HINTS = [
    "Run `nvidia-smi` — if it fails, NVIDIA drivers are not installed or not loaded.",
    "Inside a container, start it with GPU access (e.g. `docker run --gpus all ...`).",
    "Check that /dev/nvidia* device nodes exist and are readable.",
    "Ensure the `nvidia-ml-py` package is installed in this environment.",
]


def _as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class NVMLHardwareProvider:
    """Production :class:`HardwareProvider` using NVML through ``pynvml``."""

    name = "nvml"

    def __init__(self, cuda_visible_devices: list[str] | None = None) -> None:
        self._nvml: Any | None = None
        self._initialized = False
        self._cuda_visible_devices = cuda_visible_devices
        self._handles: dict[int, Any] = {}

    # ------------------------------------------------------------------ lifecycle
    def _ensure_init(self) -> Any:
        if self._initialized and self._nvml is not None:
            return self._nvml
        try:
            import pynvml
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise HardwareError(
                "ServePilot could not import the NVML bindings (nvidia-ml-py).",
                hints=["pip install nvidia-ml-py"],
            ) from exc
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as exc:
            raise HardwareError(
                "ServePilot could not access NVIDIA NVML.\n\n"
                "Possible causes:\n"
                "  1. NVIDIA drivers are not installed.\n"
                "  2. This container was not started with GPU access.\n"
                "  3. /dev/nvidia* is unavailable.\n\n"
                f"NVML error: {exc}",
                hints=NVML_INSTALL_HINTS,
            ) from exc
        self._nvml = pynvml
        self._initialized = True
        atexit.register(self.shutdown)
        return pynvml

    def shutdown(self) -> None:
        if self._initialized and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception as exc:
                log.debug("nvmlShutdown failed: %s", exc)
            self._initialized = False
            self._handles.clear()

    def _handle(self, index: int) -> Any:
        nvml = self._ensure_init()
        if index not in self._handles:
            self._handles[index] = nvml.nvmlDeviceGetHandleByIndex(index)
        return self._handles[index]

    # ------------------------------------------------------------------ queries
    def snapshot(self) -> HardwareSnapshot:
        nvml = self._ensure_init()
        try:
            count = int(nvml.nvmlDeviceGetCount())
        except nvml.NVMLError as exc:
            raise HardwareError(
                f"NVML could not enumerate GPUs: {exc}", hints=NVML_INSTALL_HINTS
            ) from exc

        gpus = [self._device(i) for i in range(count)]
        topology = self._topology(gpus) if count > 1 else GPUTopology(edges=[], available=True)
        snapshot = HardwareSnapshot(
            hostname=socket.gethostname(),
            platform=f"{platform.system()} {platform.machine()}",
            gpu_count=count,
            gpus=gpus,
            topology=topology,
            driver_version=self._driver_version(),
            cuda_version=self._cuda_version(),
            captured_at=datetime.now(tz=UTC),
            provider=self.name,
        )
        return apply_cuda_visible_devices(snapshot, self._cuda_visible_devices)

    def sample(self, gpu_indices: Sequence[int]) -> list[GPUSample]:
        nvml = self._ensure_init()
        samples: list[GPUSample] = []
        now = time.time()
        for idx in gpu_indices:
            try:
                h = self._handle(idx)
            except nvml.NVMLError as exc:
                log.debug("cannot get handle for GPU %s: %s", idx, exc)
                continue
            sample = GPUSample(index=idx, timestamp=now)
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(h)
                sample.utilization_percent = float(util.gpu)
            except nvml.NVMLError:
                pass
            try:
                mem = nvml.nvmlDeviceGetMemoryInfo(h)
                sample.memory_used_bytes = int(mem.used)
                sample.memory_total_bytes = int(mem.total)
            except nvml.NVMLError:
                pass
            with contextlib.suppress(nvml.NVMLError):
                sample.temperature_c = float(
                    nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU)
                )
            with contextlib.suppress(nvml.NVMLError):
                sample.power_watts = float(nvml.nvmlDeviceGetPowerUsage(h)) / 1000.0
            samples.append(sample)
        return samples

    # ------------------------------------------------------------------ helpers
    def _driver_version(self) -> str | None:
        nvml = self._ensure_init()
        try:
            return _as_str(nvml.nvmlSystemGetDriverVersion())
        except nvml.NVMLError:
            return None

    def _cuda_version(self) -> str | None:
        nvml = self._ensure_init()
        try:
            raw = int(nvml.nvmlSystemGetCudaDriverVersion_v2())
        except (nvml.NVMLError, AttributeError):
            try:
                raw = int(nvml.nvmlSystemGetCudaDriverVersion())
            except nvml.NVMLError:
                return None
        return f"{raw // 1000}.{(raw % 1000) // 10}"

    def _device(self, index: int) -> GPUDevice:
        nvml = self._ensure_init()
        h = self._handle(index)
        try:
            mem = nvml.nvmlDeviceGetMemoryInfo(h)
            name = _as_str(nvml.nvmlDeviceGetName(h))
            uuid = _as_str(nvml.nvmlDeviceGetUUID(h))
        except nvml.NVMLError as exc:
            raise HardwareError(f"NVML failed to query GPU {index}: {exc}") from exc

        pci_bus_id: str | None = None
        with contextlib.suppress(nvml.NVMLError):
            pci_bus_id = _as_str(nvml.nvmlDeviceGetPciInfo(h).busId)

        cc_major: int | None = None
        cc_minor: int | None = None
        with contextlib.suppress(nvml.NVMLError):
            cc_major, cc_minor = nvml.nvmlDeviceGetCudaComputeCapability(h)

        mig_mode: str | None = None
        try:
            current, _pending = nvml.nvmlDeviceGetMigMode(h)
            mig_mode = "enabled" if int(current) == 1 else "disabled"
        except nvml.NVMLError:
            mig_mode = None

        return GPUDevice(
            index=index,
            uuid=uuid,
            name=name,
            total_memory_bytes=int(mem.total),
            free_memory_bytes=int(mem.free),
            used_memory_bytes=int(mem.used),
            pci_bus_id=pci_bus_id,
            compute_capability_major=cc_major,
            compute_capability_minor=cc_minor,
            numa_node=self._numa_node(h, pci_bus_id),
            mig_mode=mig_mode,
        )

    def _numa_node(self, handle: Any, pci_bus_id: str | None) -> int | None:
        nvml = self._ensure_init()
        getter = getattr(nvml, "nvmlDeviceGetNumaNodeId", None)
        if getter is not None:
            try:
                return int(getter(handle))
            except nvml.NVMLError:
                pass
        if pci_bus_id:
            # sysfs uses lower-case, domain-truncated bus ids: 00000000:06:00.0 -> 0000:06:00.0
            bus = pci_bus_id.lower()
            if len(bus) > 12:
                bus = bus[-12:]
            path = Path("/sys/bus/pci/devices") / bus / "numa_node"
            try:
                value = int(path.read_text().strip())
                return value if value >= 0 else None
            except (OSError, ValueError):
                return None
        return None

    def _topology(self, gpus: list[GPUDevice]) -> GPUTopology:
        nvml = self._ensure_init()
        indices = [g.index for g in gpus]
        bus_to_index = {
            (g.pci_bus_id or "").lower(): g.index for g in gpus if g.pci_bus_id is not None
        }
        pcie: dict[tuple[int, int], Any] = {}
        nvlink_counts: dict[tuple[int, int], int] = {}
        nvswitch_links: dict[int, int] = {}
        any_success = False
        last_error: str | None = None

        for i, a in enumerate(indices):
            for b in indices[i + 1 :]:
                try:
                    level = int(
                        nvml.nvmlDeviceGetTopologyCommonAncestor(self._handle(a), self._handle(b))
                    )
                    pcie[(a, b)] = ancestor_to_relationship(level)
                    any_success = True
                except nvml.NVMLError as exc:
                    last_error = str(exc)

        max_links = int(getattr(nvml, "NVML_NVLINK_MAX_LINKS", 18))
        switch_type = getattr(nvml, "NVML_NVLINK_DEVICE_TYPE_SWITCH", 2)
        for a in indices:
            h = self._handle(a)
            for link in range(max_links):
                try:
                    active = int(nvml.nvmlDeviceGetNvLinkState(h, link)) == 1
                except nvml.NVMLError:
                    break  # links are contiguous; the first unsupported link ends the list
                if not active:
                    continue
                any_success = True
                remote_index: int | None = None
                remote_is_switch = False
                get_type = getattr(nvml, "nvmlDeviceGetNvLinkRemoteDeviceType", None)
                if get_type is not None:
                    try:
                        remote_is_switch = int(get_type(h, link)) == int(switch_type)
                    except nvml.NVMLError:
                        remote_is_switch = False
                if not remote_is_switch:
                    try:
                        getter = (
                            getattr(nvml, "nvmlDeviceGetNvLinkRemotePciInfo_v2", None)
                            or nvml.nvmlDeviceGetNvLinkRemotePciInfo
                        )
                        remote = getter(h, link)
                        remote_index = bus_to_index.get(_as_str(remote.busId).lower())
                    except nvml.NVMLError:
                        remote_index = None
                if remote_is_switch or remote_index is None:
                    # Remote is an NVSwitch (or unknown) - record per-GPU link count so the pair
                    # builder can treat switch-attached GPUs as fully connected.
                    nvswitch_links[a] = nvswitch_links.get(a, 0) + 1
                elif remote_index != a:
                    key = (min(a, remote_index), max(a, remote_index))
                    nvlink_counts[key] = nvlink_counts.get(key, 0) + 1

        if not any_success:
            return unavailable_topology(last_error or "NVML returned no topology information")
        # Links are counted from both ends; normalise to the per-direction count.
        for key in list(nvlink_counts):
            nvlink_counts[key] = max(1, nvlink_counts[key] // 2)
        return build_topology(indices, pcie, nvlink_counts, nvswitch_links=nvswitch_links or None)
