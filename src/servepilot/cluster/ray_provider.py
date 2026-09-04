"""Cluster-wide hardware discovery through Ray.

Each GPU node runs :class:`NVMLHardwareProvider` locally (as a Ray task pinned to that node); the
results are merged into one :class:`HardwareSnapshot` with cluster-unique GPU indices, per-node
NVLink/PCIe topology and ``NET`` edges between machines.
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from servepilot.cluster import require_ray
from servepilot.exceptions import HardwareError
from servepilot.logging import get_logger
from servepilot.schemas.hardware import (
    GPUDevice,
    GPUSample,
    GPUTopology,
    HardwareSnapshot,
    NodeInfo,
    TopologyEdge,
    TopologyRelationship,
)

log = get_logger(__name__)

NodeSnapshotFn = Callable[[str, str], HardwareSnapshot]


def _local_snapshot_payload() -> dict[str, Any]:
    """Runs on a worker node: NVML snapshot as JSON plus identity."""
    from servepilot.hardware.nvml import NVMLHardwareProvider

    snap = NVMLHardwareProvider().snapshot()
    return {"snapshot": snap.model_dump(mode="json"), "hostname": socket.gethostname()}


def _local_sample_payload(indices: list[int]) -> list[dict[str, Any]]:
    from servepilot.hardware.nvml import NVMLHardwareProvider

    return [s.model_dump(mode="json") for s in NVMLHardwareProvider().sample(indices)]


def merge_node_snapshots(
    nodes: Sequence[tuple[NodeInfo, HardwareSnapshot]], *, address: str | None
) -> HardwareSnapshot:
    """Combine per-node snapshots into a cluster snapshot (pure; unit tested with fakes)."""
    gpus: list[GPUDevice] = []
    node_infos: list[NodeInfo] = []
    edges: list[TopologyEdge] = []
    topology_available = True
    next_index = 0
    local_to_global: dict[tuple[str, int], int] = {}
    for info, snap in nodes:
        indices: list[int] = []
        for g in sorted(snap.gpus, key=lambda d: d.index):
            global_idx = next_index
            next_index += 1
            local_to_global[(info.node_id, g.index)] = global_idx
            gpus.append(
                g.model_copy(
                    update={
                        "index": global_idx,
                        "local_index": g.index,
                        "node_id": info.node_id,
                        "node_ip": info.node_ip,
                    }
                )
            )
            indices.append(global_idx)
        node_infos.append(info.model_copy(update={"gpu_indices": indices}))
        if not snap.topology.available:
            topology_available = False
        for e in snap.topology.edges:
            edges.append(
                e.model_copy(
                    update={
                        "gpu_a": local_to_global[(info.node_id, e.gpu_a)],
                        "gpu_b": local_to_global[(info.node_id, e.gpu_b)],
                    }
                )
            )
    # Cross-node edges.
    for i, a in enumerate(gpus):
        for b in gpus[i + 1 :]:
            if a.node_id != b.node_id:
                edges.append(
                    TopologyEdge(
                        gpu_a=a.index,
                        gpu_b=b.index,
                        relationship=TopologyRelationship.NETWORK.value,
                    )
                )
    edges.sort(key=lambda e: (e.gpu_a, e.gpu_b))
    head = nodes[0][1] if nodes else None
    return HardwareSnapshot(
        hostname=",".join(n.hostname or n.node_ip for n, _ in nodes),
        platform=head.platform if head else "Linux",
        gpu_count=len(gpus),
        gpus=gpus,
        topology=GPUTopology(edges=edges, available=topology_available),
        driver_version=head.driver_version if head else None,
        cuda_version=head.cuda_version if head else None,
        captured_at=datetime.now(tz=UTC),
        provider="ray",
        nodes=node_infos,
        cluster_address=address,
    )


class RayHardwareProvider:
    """A :class:`HardwareProvider` spanning every GPU node of a Ray cluster."""

    name = "ray"

    def __init__(
        self, address: str = "auto", *, node_snapshot_fn: NodeSnapshotFn | None = None
    ) -> None:
        self.address = address
        self._node_snapshot_fn = node_snapshot_fn
        self._last: HardwareSnapshot | None = None

    def _connect(self) -> Any:
        ray = require_ray()
        if not ray.is_initialized():
            try:
                ray.init(
                    address=self.address,
                    ignore_reinit_error=True,
                    log_to_driver=False,
                    logging_level=logging.WARNING,
                )
            except Exception as exc:
                raise HardwareError(
                    f"could not connect to the Ray cluster at {self.address!r}: {exc}",
                    hints=[
                        "Start a cluster with `ray start --head` on the head node and `ray start --address=<head>:6379` on workers.",
                        "Pass --ray-address auto on a node that is already part of the cluster.",
                    ],
                ) from exc
        return ray

    def _gpu_nodes(self, ray: Any) -> list[dict[str, Any]]:
        nodes = [
            n
            for n in ray.nodes()
            if n.get("Alive") and float(n.get("Resources", {}).get("GPU", 0)) > 0
        ]
        if not nodes:
            raise HardwareError(
                "the Ray cluster has no alive nodes with GPU resources.",
                hints=["Check `ray status`; workers must be started with GPUs visible."],
            )
        return sorted(
            nodes,
            key=lambda n: (
                not n.get("Resources", {}).get("node:__internal_head__"),
                n["NodeManagerAddress"],
            ),
        )

    def snapshot(self) -> HardwareSnapshot:
        ray = self._connect()
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        nodes = self._gpu_nodes(ray)
        pairs: list[tuple[NodeInfo, HardwareSnapshot]] = []
        if self._node_snapshot_fn is not None:
            for n in nodes:
                info = NodeInfo(
                    node_id=n["NodeID"],
                    node_ip=n["NodeManagerAddress"],
                    cpu_count=int(n["Resources"].get("CPU", 0)),
                )
                pairs.append((info, self._node_snapshot_fn(info.node_id, info.node_ip)))
        else:
            task = ray.remote(num_cpus=0)(_local_snapshot_payload)
            futures = [
                task.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=n["NodeID"], soft=False
                    )
                ).remote()
                for n in nodes
            ]
            try:
                payloads = ray.get(futures, timeout=180)
            except Exception as exc:
                raise HardwareError(
                    f"hardware inspection failed on a Ray worker: {exc}",
                    hints=["Every worker needs ServePilot and NVIDIA drivers installed."],
                ) from exc
            for n, payload in zip(nodes, payloads, strict=True):
                resources = n.get("Resources", {})
                info = NodeInfo(
                    node_id=n["NodeID"],
                    node_ip=n["NodeManagerAddress"],
                    hostname=payload.get("hostname"),
                    cpu_count=int(resources.get("CPU", 0)) or None,
                    memory_bytes=int(resources.get("memory", 0)) or None,
                    is_head=bool(resources.get("node:__internal_head__")),
                )
                pairs.append((info, HardwareSnapshot.model_validate(payload["snapshot"])))
        self._last = merge_node_snapshots(pairs, address=self.address)
        return self._last

    def sample(self, gpu_indices: Sequence[int]) -> list[GPUSample]:
        if self._last is None:
            self.snapshot()
        assert self._last is not None
        ray = self._connect()
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        by_node: dict[str, list[GPUDevice]] = {}
        for idx in gpu_indices:
            try:
                g = self._last.gpu(idx)
            except KeyError:
                continue
            if g.node_id:
                by_node.setdefault(g.node_id, []).append(g)
        task = ray.remote(num_cpus=0)(_local_sample_payload)
        futures = {
            node: task.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node, soft=False)
            ).remote([g.device_index_on_node for g in devices])
            for node, devices in by_node.items()
        }
        out: list[GPUSample] = []
        for node, fut in futures.items():
            try:
                payload = ray.get(fut, timeout=30)
            except Exception as exc:
                log.debug("sampling failed on node %s: %s", node, exc)
                continue
            local_to_global = {g.device_index_on_node: g.index for g in by_node[node]}
            for raw in payload:
                sample = GPUSample.model_validate(raw)
                sample.index = local_to_global.get(sample.index, sample.index)
                out.append(sample)
        return out
