"""Hardware layer: topology building, NVML normalisation (mocked), visibility, fingerprints."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from servepilot.constants import GIB
from servepilot.exceptions import HardwareError
from servepilot.hardware.base import apply_cuda_visible_devices, parse_cuda_visible_devices
from servepilot.hardware.fingerprint import hardware_fingerprint, hardware_identity
from servepilot.hardware.nvml import NVMLHardwareProvider
from servepilot.hardware.topology import (
    NVML_TOPOLOGY_HOSTBRIDGE,
    NVML_TOPOLOGY_SYSTEM,
    ancestor_to_relationship,
    build_topology,
    unavailable_topology,
)
from servepilot.schemas.hardware import TopologyRelationship
from servepilot.testing import fake_hardware as fh


class TestTopology:
    def test_ancestor_mapping(self) -> None:
        assert ancestor_to_relationship(NVML_TOPOLOGY_HOSTBRIDGE) == TopologyRelationship.PHB
        assert ancestor_to_relationship(NVML_TOPOLOGY_SYSTEM) == TopologyRelationship.SYS
        assert ancestor_to_relationship(None) == TopologyRelationship.UNKNOWN
        assert ancestor_to_relationship(999) == TopologyRelationship.UNKNOWN

    def test_build_topology_marks_nvlink_pairs(self) -> None:
        topo = build_topology(
            [0, 1, 2],
            {
                (0, 1): TopologyRelationship.PHB,
                (0, 2): TopologyRelationship.SYS,
                (1, 2): TopologyRelationship.SYS,
            },
            {(0, 1): 12},
        )
        e01 = topo.edge(0, 1)
        assert e01 is not None and e01.nvlink_detected and e01.nvlink_link_count == 12
        assert e01.relationship == "NV"
        e02 = topo.edge(2, 0)  # order-insensitive lookup
        assert e02 is not None and not e02.nvlink_detected and e02.relationship == "SYS"
        assert topo.score(0, 1) > topo.score(0, 2)
        assert topo.summary() == "partial NVLink (1/3 pairs)"

    def test_nvswitch_links_imply_full_connectivity(self) -> None:
        topo = build_topology([0, 1, 2, 3], {}, {}, nvswitch_links={0: 18, 1: 18, 2: 18, 3: 18})
        assert all(e.nvlink_detected and e.nvlink_link_count == 18 for e in topo.edges)
        assert topo.summary() == "full NVLink mesh"

    def test_unavailable_topology(self) -> None:
        topo = unavailable_topology("boom")
        assert not topo.available and topo.edges == [] and topo.error == "boom"
        assert topo.summary() == "topology unavailable"
        assert topo.score(0, 1) == 1  # unknown score, never crashes


class TestVisibility:
    def test_parse(self) -> None:
        assert parse_cuda_visible_devices(None) is None
        assert parse_cuda_visible_devices("0, 2") == ["0", "2"]
        assert parse_cuda_visible_devices("") == []

    def test_apply_indices_and_uuids(self) -> None:
        snap = fh.h100x4()
        restricted = apply_cuda_visible_devices(snap, ["1", snap.gpus[3].uuid, "99"])
        assert [g.index for g in restricted.gpus] == [1, 3]
        assert restricted.gpu_count == 2
        assert all({e.gpu_a, e.gpu_b} <= {1, 3} for e in restricted.topology.edges)

    def test_empty_visibility_hides_everything(self) -> None:
        restricted = apply_cuda_visible_devices(fh.h100x4(), [])
        assert restricted.gpu_count == 0


class TestFixtures:
    def test_mixed_detection(self) -> None:
        snap = fh.mixed_h100_a100()
        assert not snap.is_homogeneous
        groups = snap.homogeneous_groups()
        assert sorted(len(v) for v in groups.values()) == [2, 2]

    def test_partially_occupied(self) -> None:
        snap = fh.partially_occupied_x4()
        busy = [g for g in snap.gpus if g.used_fraction > 0.05]
        assert [g.index for g in busy] == [1]
        assert snap.min_free_memory_bytes < snap.min_total_memory_bytes - 29 * GIB

    def test_cluster_fixture(self) -> None:
        snap = fh.cluster(2, 4)
        assert snap.is_cluster and snap.gpu_count == 8
        assert snap.node_of(5) == "node-1"
        assert snap.gpu(5).device_index_on_node == 1
        cross = snap.topology.edge(0, 4)
        assert cross is not None and cross.relationship == "NET"
        intra = snap.topology.edge(4, 5)
        assert intra is not None and intra.nvlink_detected
        sub = snap.select([4, 5, 6, 7])
        assert [n.node_id for n in sub.nodes] == ["node-1"]

    def test_fixture_registry(self) -> None:
        assert fh.fixture_by_name("h100x8").gpu_count == 8
        with pytest.raises(KeyError):
            fh.fixture_by_name("nope")


class TestFingerprint:
    def test_stable_across_volatile_fields(self) -> None:
        a = fh.h100x8()
        b = fh.h100x8()
        for g in b.gpus:
            g.free_memory_bytes -= 5 * GIB
            g.used_memory_bytes += 5 * GIB
        b.hostname = "other-host"
        assert hardware_fingerprint(a) == hardware_fingerprint(b)

    def test_changes_with_topology_and_count(self) -> None:
        base = hardware_fingerprint(fh.h100x4())
        assert hardware_fingerprint(fh.h100(4, nvlink=False)) != base
        assert hardware_fingerprint(fh.h100x8()) != base
        assert hardware_fingerprint(fh.h100x8(), gpu_ids=[0, 1, 2, 3]) == base

    def test_identity_contents(self) -> None:
        ident = hardware_identity(fh.h100x1())
        assert ident["gpus"][0]["name"].startswith("NVIDIA H100")  # type: ignore[index]
        assert "free_memory_bytes" not in str(ident)


# ----------------------------------------------------------------------------- NVML mock
class _Mem:
    def __init__(self, total: int, free: int) -> None:
        self.total, self.free, self.used = total, free, total - free


class _Pci:
    def __init__(self, bus: str) -> None:
        self.busId = bus.encode()


class _NVMLError(Exception):
    pass


def _fake_pynvml(
    gpu_count: int = 2, *, fail_init: bool = False, topology_fails: bool = False
) -> types.ModuleType:
    mod = types.ModuleType("pynvml")
    mod.NVMLError = _NVMLError  # type: ignore[attr-defined]
    mod.NVML_NVLINK_MAX_LINKS = 18  # type: ignore[attr-defined]
    mod.NVML_TEMPERATURE_GPU = 0  # type: ignore[attr-defined]
    buses = [f"00000000:{6 + i:02X}:00.0" for i in range(gpu_count)]

    def nvmlInit() -> None:
        if fail_init:
            raise _NVMLError("Driver Not Loaded")

    mod.nvmlInit = nvmlInit  # type: ignore[attr-defined]
    mod.nvmlShutdown = lambda: None  # type: ignore[attr-defined]
    mod.nvmlDeviceGetCount = lambda: gpu_count  # type: ignore[attr-defined]
    mod.nvmlDeviceGetHandleByIndex = lambda i: ("handle", i)  # type: ignore[attr-defined]
    mod.nvmlSystemGetDriverVersion = lambda: b"580.65.06"  # type: ignore[attr-defined]
    mod.nvmlSystemGetCudaDriverVersion_v2 = lambda: 13000  # type: ignore[attr-defined]
    mod.nvmlDeviceGetMemoryInfo = lambda h: _Mem(80 * GIB, 79 * GIB)  # type: ignore[attr-defined]
    mod.nvmlDeviceGetName = lambda h: b"NVIDIA H100 80GB HBM3"  # type: ignore[attr-defined]
    mod.nvmlDeviceGetUUID = lambda h: f"GPU-{h[1]:032d}"  # type: ignore[attr-defined]
    mod.nvmlDeviceGetPciInfo = lambda h: _Pci(buses[h[1]])  # type: ignore[attr-defined]
    mod.nvmlDeviceGetCudaComputeCapability = lambda h: (9, 0)  # type: ignore[attr-defined]

    def mig(h: Any) -> Any:
        raise _NVMLError("Not Supported")

    mod.nvmlDeviceGetMigMode = mig  # type: ignore[attr-defined]

    def ancestor(a: Any, b: Any) -> int:
        if topology_fails:
            raise _NVMLError("Not Supported")
        return NVML_TOPOLOGY_HOSTBRIDGE

    mod.nvmlDeviceGetTopologyCommonAncestor = ancestor  # type: ignore[attr-defined]

    def link_state(h: Any, link: int) -> int:
        if topology_fails:
            raise _NVMLError("Not Supported")
        if link >= 4:
            raise _NVMLError("Invalid Argument")
        return 1

    mod.nvmlDeviceGetNvLinkState = link_state  # type: ignore[attr-defined]
    mod.nvmlDeviceGetNvLinkRemoteDeviceType = lambda h, link: 0  # type: ignore[attr-defined]

    def remote(h: Any, link: int) -> Any:
        other = 1 - h[1] if gpu_count == 2 else (h[1] + 1) % gpu_count
        return _Pci(buses[other])

    mod.nvmlDeviceGetNvLinkRemotePciInfo_v2 = remote  # type: ignore[attr-defined]
    mod.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=42, memory=10)  # type: ignore[attr-defined]
    mod.nvmlDeviceGetTemperature = lambda h, kind: 55  # type: ignore[attr-defined]
    mod.nvmlDeviceGetPowerUsage = lambda h: 300_000  # type: ignore[attr-defined]
    return mod


class TestNVMLProvider:
    def test_snapshot_normalisation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(2))
        provider = NVMLHardwareProvider()
        snap = provider.snapshot()
        assert snap.gpu_count == 2
        assert snap.driver_version == "580.65.06" and snap.cuda_version == "13.0"
        g = snap.gpus[0]
        assert g.name == "NVIDIA H100 80GB HBM3" and g.total_memory_bytes == 80 * GIB
        assert g.compute_capability == "9.0" and g.mig_mode is None
        assert g.pci_bus_id == "00000000:06:00.0"
        edge = snap.topology.edge(0, 1)
        assert edge is not None and edge.nvlink_detected and edge.nvlink_link_count == 4
        samples = provider.sample([0, 1])
        assert (
            len(samples) == 2
            and samples[0].utilization_percent == 42
            and samples[0].power_watts == 300.0
        )
        provider.shutdown()

    def test_topology_failure_degrades_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(2, topology_fails=True))
        snap = NVMLHardwareProvider().snapshot()
        assert snap.gpu_count == 2
        assert not snap.topology.available

    def test_init_failure_is_actionable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(2, fail_init=True))
        with pytest.raises(HardwareError) as exc:
            NVMLHardwareProvider().snapshot()
        assert "nvidia-smi" in exc.value.render()

    def test_respects_cuda_visible_devices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(2))
        snap = NVMLHardwareProvider(cuda_visible_devices=["1"]).snapshot()
        assert [g.index for g in snap.gpus] == [1]
