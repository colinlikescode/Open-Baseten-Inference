"""Hardware snapshot schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class TopologyRelationship(StrEnum):
    """Interconnect relationship between two GPUs, mirroring ``nvidia-smi topo`` vocabulary.

    Ordered from best to worst locality. ``NVLINK`` is used when a direct NVLink connection is
    detected regardless of the PCIe ancestor.
    """

    NVLINK = "NV"
    PIX = "PIX"  # same PCIe switch
    PXB = "PXB"  # multiple PCIe bridges, no host bridge
    PHB = "PHB"  # same host bridge / CPU root complex
    NODE = "NODE"  # same NUMA node, different host bridges
    SYS = "SYS"  # cross NUMA / QPI / UPI
    NETWORK = "NET"  # different machines (Ray cluster)
    UNKNOWN = "UNKNOWN"


# Higher is better. Used by the GPU group planner to weigh pairs.
RELATIONSHIP_SCORES: dict[TopologyRelationship, int] = {
    TopologyRelationship.NVLINK: 100,
    TopologyRelationship.PIX: 40,
    TopologyRelationship.PXB: 30,
    TopologyRelationship.PHB: 20,
    TopologyRelationship.NODE: 10,
    TopologyRelationship.SYS: 5,
    TopologyRelationship.UNKNOWN: 1,
    TopologyRelationship.NETWORK: 0,
}


class GPUDevice(BaseModel):
    """A single NVIDIA GPU as observed through NVML (or a fake provider).

    ``index`` is unique across the whole snapshot. On a single machine it equals the NVML index.
    In a Ray cluster indices are assigned sequentially across nodes and ``local_index`` holds the
    NVML index on ``node_id`` (used for ``CUDA_VISIBLE_DEVICES`` on that machine).
    """

    index: int
    uuid: str
    name: str
    total_memory_bytes: int
    free_memory_bytes: int
    used_memory_bytes: int
    pci_bus_id: str | None = None
    compute_capability_major: int | None = None
    compute_capability_minor: int | None = None
    numa_node: int | None = None
    mig_mode: str | None = None
    node_id: str | None = None
    node_ip: str | None = None
    local_index: int | None = None

    @property
    def device_index_on_node(self) -> int:
        return self.local_index if self.local_index is not None else self.index

    @property
    def compute_capability(self) -> str | None:
        if self.compute_capability_major is None:
            return None
        minor = self.compute_capability_minor if self.compute_capability_minor is not None else 0
        return f"{self.compute_capability_major}.{minor}"

    @property
    def used_fraction(self) -> float:
        if self.total_memory_bytes <= 0:
            return 0.0
        return self.used_memory_bytes / self.total_memory_bytes


class TopologyEdge(BaseModel):
    """Relationship between a pair of GPUs (``gpu_a < gpu_b``)."""

    gpu_a: int
    gpu_b: int
    relationship: str = TopologyRelationship.UNKNOWN.value
    nvlink_detected: bool = False
    nvlink_link_count: int | None = None

    def score(self) -> int:
        """Connectivity score used for grouping; more NVLink links score higher."""
        try:
            rel = TopologyRelationship(self.relationship)
        except ValueError:
            rel = TopologyRelationship.UNKNOWN
        base = RELATIONSHIP_SCORES[rel]
        if self.nvlink_detected:
            base = max(base, RELATIONSHIP_SCORES[TopologyRelationship.NVLINK])
            base += self.nvlink_link_count or 0
        return base


class GPUTopology(BaseModel):
    """Pairwise GPU interconnect information.

    ``available`` is False when topology could not be queried at all; edges are then empty and
    planners must fall back to index-based grouping.
    """

    edges: list[TopologyEdge] = Field(default_factory=list)
    available: bool = True
    error: str | None = None

    def edge(self, a: int, b: int) -> TopologyEdge | None:
        lo, hi = (a, b) if a < b else (b, a)
        for e in self.edges:
            if e.gpu_a == lo and e.gpu_b == hi:
                return e
        return None

    def score(self, a: int, b: int) -> int:
        e = self.edge(a, b)
        return e.score() if e is not None else RELATIONSHIP_SCORES[TopologyRelationship.UNKNOWN]

    def has_nvlink(self) -> bool:
        return any(e.nvlink_detected for e in self.edges)

    def summary(self) -> str:
        if not self.available:
            return "topology unavailable"
        if not self.edges:
            return "single GPU"
        nv = sum(1 for e in self.edges if e.nvlink_detected)
        if nv == len(self.edges):
            return "full NVLink mesh"
        if nv > 0:
            return f"partial NVLink ({nv}/{len(self.edges)} pairs)"
        rels = sorted({e.relationship for e in self.edges})
        return "PCIe only (" + ", ".join(rels) + ")"


class NodeInfo(BaseModel):
    """One machine in a (possibly single-node) deployment."""

    node_id: str
    node_ip: str
    hostname: str | None = None
    gpu_indices: list[int] = Field(default_factory=list)
    cpu_count: int | None = None
    memory_bytes: int | None = None
    is_head: bool = False


class HardwareSnapshot(BaseModel):
    """Everything ServePilot knows about the machine (or Ray cluster) at one point in time."""

    hostname: str
    platform: str
    gpu_count: int
    gpus: list[GPUDevice] = Field(default_factory=list)
    topology: GPUTopology = Field(default_factory=GPUTopology)
    driver_version: str | None = None
    cuda_version: str | None = None
    captured_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    provider: str = "nvml"
    nodes: list[NodeInfo] = Field(default_factory=list)
    cluster_address: str | None = None

    @property
    def is_cluster(self) -> bool:
        return len(self.nodes) > 1

    def gpu(self, index: int) -> GPUDevice:
        for g in self.gpus:
            if g.index == index:
                return g
        raise KeyError(f"GPU index {index} not present in snapshot")

    def node_of(self, index: int) -> str | None:
        return self.gpu(index).node_id

    def node(self, node_id: str) -> NodeInfo | None:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def gpus_by_node(self) -> dict[str | None, list[int]]:
        out: dict[str | None, list[int]] = {}
        for g in self.gpus:
            out.setdefault(g.node_id, []).append(g.index)
        return out

    def select(self, indices: list[int]) -> HardwareSnapshot:
        """Return a snapshot restricted to ``indices`` (keeps original indices)."""
        wanted = set(indices)
        gpus = [g for g in self.gpus if g.index in wanted]
        edges = [e for e in self.topology.edges if e.gpu_a in wanted and e.gpu_b in wanted]
        nodes = [
            n.model_copy(update={"gpu_indices": [i for i in n.gpu_indices if i in wanted]})
            for n in self.nodes
        ]
        nodes = [n for n in nodes if n.gpu_indices or not self.nodes]
        return self.model_copy(
            update={
                "gpus": gpus,
                "gpu_count": len(gpus),
                "topology": self.topology.model_copy(update={"edges": edges}),
                "nodes": nodes,
            }
        )

    def homogeneous_groups(self) -> dict[tuple[str, int], list[int]]:
        """Group GPU indices by (name, total memory)."""
        groups: dict[tuple[str, int], list[int]] = {}
        for g in self.gpus:
            groups.setdefault((g.name, g.total_memory_bytes), []).append(g.index)
        return groups

    @property
    def is_homogeneous(self) -> bool:
        return len(self.homogeneous_groups()) <= 1

    @property
    def min_total_memory_bytes(self) -> int:
        return min((g.total_memory_bytes for g in self.gpus), default=0)

    @property
    def min_free_memory_bytes(self) -> int:
        return min((g.free_memory_bytes for g in self.gpus), default=0)


class GPUSample(BaseModel):
    """A point-in-time utilization sample for one GPU."""

    index: int
    timestamp: float
    utilization_percent: float | None = None
    memory_used_bytes: int | None = None
    memory_total_bytes: int | None = None
    temperature_c: float | None = None
    power_watts: float | None = None
