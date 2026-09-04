"""Fake hardware provider and reusable machine fixtures.

Fixture names are test labels only; memory sizes approximate real products so planner behaviour
is realistic, but nothing here is a hardware lookup table used by production code.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from servepilot.constants import GIB
from servepilot.hardware.topology import build_topology, unavailable_topology
from servepilot.schemas.hardware import (
    GPUDevice,
    GPUSample,
    GPUTopology,
    HardwareSnapshot,
    NodeInfo,
    TopologyRelationship,
)


class FakeHardwareProvider:
    """A :class:`HardwareProvider` that serves a fixed snapshot and synthetic samples."""

    name = "fake"

    def __init__(self, snapshot: HardwareSnapshot) -> None:
        self._snapshot = snapshot
        self.snapshot_calls = 0
        self.utilization_percent = 75.0
        self.power_watts = 300.0

    def snapshot(self) -> HardwareSnapshot:
        self.snapshot_calls += 1
        return self._snapshot.model_copy(update={"captured_at": datetime.now(tz=UTC)}, deep=True)

    def sample(self, gpu_indices: Sequence[int]) -> list[GPUSample]:
        now = time.time()
        out: list[GPUSample] = []
        for idx in gpu_indices:
            try:
                g = self._snapshot.gpu(idx)
            except KeyError:
                continue
            out.append(
                GPUSample(
                    index=idx,
                    timestamp=now,
                    utilization_percent=self.utilization_percent,
                    memory_used_bytes=g.used_memory_bytes,
                    memory_total_bytes=g.total_memory_bytes,
                    temperature_c=55.0,
                    power_watts=self.power_watts,
                )
            )
        return out

    def set_free_memory(self, index: int, free_bytes: int) -> None:
        """Mutate free/used memory of one GPU (simulates other processes)."""
        for g in self._snapshot.gpus:
            if g.index == index:
                g.free_memory_bytes = free_bytes
                g.used_memory_bytes = g.total_memory_bytes - free_bytes


def make_gpu(
    index: int,
    name: str,
    total_bytes: int,
    *,
    used_bytes: int = 0,
    compute_capability: tuple[int, int] = (9, 0),
    numa_node: int | None = 0,
) -> GPUDevice:
    return GPUDevice(
        index=index,
        uuid=f"GPU-fake-{name.replace(' ', '-').lower()}-{index:02d}",
        name=name,
        total_memory_bytes=total_bytes,
        free_memory_bytes=total_bytes - used_bytes,
        used_memory_bytes=used_bytes,
        pci_bus_id=f"00000000:{index + 6:02X}:00.0",
        compute_capability_major=compute_capability[0],
        compute_capability_minor=compute_capability[1],
        numa_node=numa_node,
        mig_mode="disabled",
    )


def full_nvlink_topology(indices: Sequence[int], links: int = 18) -> GPUTopology:
    pcie = {
        (a, b): TopologyRelationship.PHB for i, a in enumerate(indices) for b in indices[i + 1 :]
    }
    nv = {(a, b): links for i, a in enumerate(indices) for b in indices[i + 1 :]}
    return build_topology(indices, pcie, nv)


def pcie_only_topology(
    indices: Sequence[int], relationship: TopologyRelationship = TopologyRelationship.PHB
) -> GPUTopology:
    pcie = {(a, b): relationship for i, a in enumerate(indices) for b in indices[i + 1 :]}
    return build_topology(indices, pcie, {})


def grouped_nvlink_topology(indices: Sequence[int], group_size: int, links: int = 4) -> GPUTopology:
    """NVLink inside consecutive groups of ``group_size``; PCIe (SYS) across groups."""
    pcie: dict[tuple[int, int], TopologyRelationship] = {}
    nv: dict[tuple[int, int], int] = {}
    for i, a in enumerate(indices):
        for b in indices[i + 1 :]:
            same_group = (a // group_size) == (b // group_size)
            pcie[(a, b)] = TopologyRelationship.PXB if same_group else TopologyRelationship.SYS
            if same_group:
                nv[(a, b)] = links
    return build_topology(indices, pcie, nv)


def make_snapshot(
    gpus: list[GPUDevice], topology: GPUTopology, *, driver: str = "580.65.06", cuda: str = "13.0"
) -> HardwareSnapshot:
    return HardwareSnapshot(
        hostname="fake-host",
        platform="Linux x86_64",
        gpu_count=len(gpus),
        gpus=gpus,
        topology=topology,
        driver_version=driver,
        cuda_version=cuda,
        captured_at=datetime.now(tz=UTC),
        provider="fake",
    )


H100_NAME = "NVIDIA H100 80GB HBM3"
H100_BYTES = 85_520_809_984  # 81559 MiB as reported by nvidia-smi
A100_NAME = "NVIDIA A100-SXM4-80GB"
A100_BYTES = 85_899_345_920
A100_40_NAME = "NVIDIA A100-SXM4-40GB"
A100_40_BYTES = 42_949_672_960
B200_NAME = "NVIDIA B200"
B200_BYTES = 193_273_528_320  # ~180 GiB
RTX4090_NAME = "NVIDIA GeForce RTX 4090"
RTX4090_BYTES = 25_757_220_864


def h100(count: int, *, nvlink: bool = True, used_bytes: int = 0) -> HardwareSnapshot:
    gpus = [make_gpu(i, H100_NAME, H100_BYTES, used_bytes=used_bytes) for i in range(count)]
    idx = list(range(count))
    topo = full_nvlink_topology(idx) if nvlink else pcie_only_topology(idx)
    if count == 1:
        topo = GPUTopology(edges=[], available=True)
    return make_snapshot(gpus, topo)


def h100x1() -> HardwareSnapshot:
    return h100(1)


def h100x2() -> HardwareSnapshot:
    return h100(2)


def h100x4() -> HardwareSnapshot:
    return h100(4)


def h100x8() -> HardwareSnapshot:
    return h100(8)


def b200x8() -> HardwareSnapshot:
    gpus = [make_gpu(i, B200_NAME, B200_BYTES, compute_capability=(10, 0)) for i in range(8)]
    return make_snapshot(gpus, full_nvlink_topology(list(range(8))))


def a100x8() -> HardwareSnapshot:
    gpus = [make_gpu(i, A100_NAME, A100_BYTES, compute_capability=(8, 0)) for i in range(8)]
    return make_snapshot(
        gpus, full_nvlink_topology(list(range(8)), links=12), driver="550.90.07", cuda="12.4"
    )


def mixed_h100_a100() -> HardwareSnapshot:
    gpus = [
        make_gpu(0, H100_NAME, H100_BYTES),
        make_gpu(1, H100_NAME, H100_BYTES),
        make_gpu(2, A100_40_NAME, A100_40_BYTES, compute_capability=(8, 0), numa_node=1),
        make_gpu(3, A100_40_NAME, A100_40_BYTES, compute_capability=(8, 0), numa_node=1),
    ]
    topo = grouped_nvlink_topology([0, 1, 2, 3], group_size=2, links=12)
    return make_snapshot(gpus, topo)


def no_nvlink_x4() -> HardwareSnapshot:
    gpus = [make_gpu(i, RTX4090_NAME, RTX4090_BYTES, compute_capability=(8, 9)) for i in range(4)]
    return make_snapshot(gpus, pcie_only_topology(list(range(4)), TopologyRelationship.PHB))


def nvlink_groups_x8() -> HardwareSnapshot:
    """Two NVLink islands of four GPUs (0-3, 4-7) connected through the host."""
    gpus = [
        make_gpu(i, A100_NAME, A100_BYTES, compute_capability=(8, 0), numa_node=i // 4)
        for i in range(8)
    ]
    return make_snapshot(gpus, grouped_nvlink_topology(list(range(8)), group_size=4, links=4))


def nvlink_pairs_x4() -> HardwareSnapshot:
    """NVLink bridges between (0,1) and (2,3) only."""
    gpus = [make_gpu(i, A100_NAME, A100_BYTES, compute_capability=(8, 0)) for i in range(4)]
    return make_snapshot(gpus, grouped_nvlink_topology(list(range(4)), group_size=2, links=12))


def scattered_nvlink_x4() -> HardwareSnapshot:
    """NVLink between (0,2) and (1,3): the best pairing is *not* consecutive indices."""
    idx = [0, 1, 2, 3]
    pcie = {(a, b): TopologyRelationship.SYS for i, a in enumerate(idx) for b in idx[i + 1 :]}
    nv = {(0, 2): 12, (1, 3): 12}
    gpus = [make_gpu(i, A100_NAME, A100_BYTES, compute_capability=(8, 0)) for i in idx]
    return make_snapshot(gpus, build_topology(idx, pcie, nv))


def partially_occupied_x4() -> HardwareSnapshot:
    """GPU 1 has 30 GiB used by another process."""
    gpus = [
        make_gpu(i, H100_NAME, H100_BYTES, used_bytes=(30 * GIB if i == 1 else 0)) for i in range(4)
    ]
    return make_snapshot(gpus, full_nvlink_topology(list(range(4))))


def topology_unavailable_x4() -> HardwareSnapshot:
    gpus = [make_gpu(i, H100_NAME, H100_BYTES) for i in range(4)]
    return make_snapshot(gpus, unavailable_topology("simulated NVML topology failure"))


def cluster(
    node_count: int,
    gpus_per_node: int,
    *,
    name: str = H100_NAME,
    total_bytes: int = H100_BYTES,
    nvlink: bool = True,
) -> HardwareSnapshot:
    """A Ray-style multi-node snapshot: NVLink inside each node, network between nodes."""
    gpus: list[GPUDevice] = []
    nodes: list[NodeInfo] = []
    pcie: dict[tuple[int, int], TopologyRelationship] = {}
    nv: dict[tuple[int, int], int] = {}
    for n in range(node_count):
        node_id = f"node-{n}"
        node_ip = f"10.0.0.{n + 1}"
        indices: list[int] = []
        for local in range(gpus_per_node):
            idx = n * gpus_per_node + local
            g = make_gpu(local, name, total_bytes)
            g.index = idx
            g.local_index = local
            g.node_id = node_id
            g.node_ip = node_ip
            g.uuid = f"GPU-fake-{node_id}-{local:02d}"
            gpus.append(g)
            indices.append(idx)
        nodes.append(
            NodeInfo(
                node_id=node_id,
                node_ip=node_ip,
                hostname=f"worker-{n}",
                gpu_indices=indices,
                cpu_count=64,
                memory_bytes=512 * GIB,
                is_head=(n == 0),
            )
        )
    all_idx = [g.index for g in gpus]
    for i, a in enumerate(all_idx):
        for b in all_idx[i + 1 :]:
            same_node = gpus[a].node_id == gpus[b].node_id
            if same_node:
                pcie[(a, b)] = TopologyRelationship.PHB
                if nvlink:
                    nv[(a, b)] = 18
            else:
                pcie[(a, b)] = TopologyRelationship.NETWORK
    snap = make_snapshot(gpus, build_topology(all_idx, pcie, nv))
    snap.nodes = nodes
    snap.cluster_address = "ray://10.0.0.1:10001"
    snap.provider = "ray"
    return snap


def cluster_2x4() -> HardwareSnapshot:
    return cluster(2, 4)


def cluster_2x8() -> HardwareSnapshot:
    return cluster(2, 8)


FIXTURES: dict[str, Callable[[], HardwareSnapshot]] = {
    "cluster_2x4": cluster_2x4,
    "cluster_2x8": cluster_2x8,
    "h100x1": h100x1,
    "h100x2": h100x2,
    "h100x4": h100x4,
    "h100x8": h100x8,
    "b200x8": b200x8,
    "a100x8": a100x8,
    "mixed_h100_a100": mixed_h100_a100,
    "no_nvlink_x4": no_nvlink_x4,
    "nvlink_groups_x8": nvlink_groups_x8,
    "nvlink_pairs_x4": nvlink_pairs_x4,
    "scattered_nvlink_x4": scattered_nvlink_x4,
    "partially_occupied_x4": partially_occupied_x4,
    "topology_unavailable_x4": topology_unavailable_x4,
}


def fixture_by_name(name: str) -> HardwareSnapshot:
    try:
        return FIXTURES[name]()
    except KeyError as exc:
        raise KeyError(
            f"unknown fake hardware fixture {name!r}; choose from {sorted(FIXTURES)}"
        ) from exc
