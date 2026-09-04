"""Cluster snapshot merging and cluster-aware planning (no Ray needed)."""

from __future__ import annotations

from servepilot.cluster.ray_provider import merge_node_snapshots
from servepilot.hardware.fingerprint import hardware_fingerprint
from servepilot.schemas.hardware import NodeInfo
from servepilot.testing import fake_hardware as fh


def test_merge_node_snapshots_assigns_global_indices_and_network_edges() -> None:
    a = fh.h100x2()
    b = fh.h100x2()
    merged = merge_node_snapshots(
        [
            (NodeInfo(node_id="n0", node_ip="10.0.0.1", is_head=True), a),
            (NodeInfo(node_id="n1", node_ip="10.0.0.2"), b),
        ],
        address="auto",
    )
    assert merged.is_cluster and merged.gpu_count == 4 and merged.provider == "ray"
    assert [g.index for g in merged.gpus] == [0, 1, 2, 3]
    assert (
        merged.gpu(2).node_id == "n1"
        and merged.gpu(2).local_index == 0
        and merged.gpu(2).node_ip == "10.0.0.2"
    )
    assert merged.nodes[1].gpu_indices == [2, 3]
    intra = merged.topology.edge(2, 3)
    cross = merged.topology.edge(1, 2)
    assert intra is not None and intra.nvlink_detected
    assert cross is not None and cross.relationship == "NET" and not cross.nvlink_detected
    assert merged.driver_version == a.driver_version and merged.cluster_address == "auto"
    assert merged.gpus_by_node() == {"n0": [0, 1], "n1": [2, 3]}


def test_merge_propagates_topology_unavailability() -> None:
    merged = merge_node_snapshots(
        [
            (NodeInfo(node_id="n0", node_ip="10.0.0.1"), fh.topology_unavailable_x4()),
            (NodeInfo(node_id="n1", node_ip="10.0.0.2"), fh.h100x4()),
        ],
        address=None,
    )
    assert not merged.topology.available and merged.gpu_count == 8


def test_cluster_fingerprint_depends_on_node_count() -> None:
    assert hardware_fingerprint(fh.cluster(2, 4)) != hardware_fingerprint(fh.cluster(1, 4))
    assert hardware_fingerprint(fh.cluster(2, 4)) == hardware_fingerprint(fh.cluster(2, 4))
