"""GPU group planning against fake topologies."""

from __future__ import annotations

import pytest

from servepilot.exceptions import NoViablePlanError
from servepilot.hardware.topology import build_topology
from servepilot.planner.gpu_groups import GPUGroupPlanner
from servepilot.schemas.hardware import TopologyRelationship
from servepilot.testing import fake_hardware as fh


class TestGrouping:
    def test_tp1_is_trivial(self) -> None:
        planner = GPUGroupPlanner(fh.h100x4())
        assert planner.plan(1, 4) == [[0], [1], [2], [3]]
        assert planner.plan(1, 2, [3, 1]) == [[1], [3]]

    def test_prefers_direct_nvlink_over_index_order(self) -> None:
        planner = GPUGroupPlanner(fh.scattered_nvlink_x4())  # NVLink between (0,2) and (1,3)
        assert planner.plan(2, 2) == [[0, 2], [1, 3]]

    def test_nvlink_pairs_fixture(self) -> None:
        planner = GPUGroupPlanner(fh.nvlink_pairs_x4())
        assert planner.plan(2, 2) == [[0, 1], [2, 3]]
        assert planner.plan(4, 1) == [[0, 1, 2, 3]]

    def test_nvlink_islands_of_four(self) -> None:
        planner = GPUGroupPlanner(fh.nvlink_groups_x8())
        assert planner.plan(4, 2) == [[0, 1, 2, 3], [4, 5, 6, 7]]
        pairs = planner.plan(2, 4)
        for group in pairs:
            assert group[0] // 4 == group[1] // 4, "pairs must not cross NVLink islands"

    def test_disconnected_topology_falls_back_to_adjacent_indices(self) -> None:
        planner = GPUGroupPlanner(fh.topology_unavailable_x4())
        assert planner.plan(2, 2) == [[0, 1], [2, 3]]

    def test_pcie_only_deterministic_ties(self) -> None:
        planner = GPUGroupPlanner(fh.no_nvlink_x4())
        first = planner.plan(2, 2)
        assert first == planner.plan(2, 2) == [[0, 1], [2, 3]]

    def test_leftover_gpus_pick_best_subset(self) -> None:
        # 4 GPUs, one TP=2 group requested: choose the NVLink-connected pair (1,3), not (0,1).
        idx = [0, 1, 2, 3]
        pcie = {(a, b): TopologyRelationship.SYS for i, a in enumerate(idx) for b in idx[i + 1 :]}
        snap = fh.scattered_nvlink_x4()
        snap.topology = build_topology(idx, pcie, {(1, 3): 12})
        planner = GPUGroupPlanner(snap)
        assert planner.plan(2, 1) == [[1, 3]]

    def test_greedy_path_for_large_machines(self) -> None:
        snap = fh.cluster(1, 16)
        for g in snap.gpus:
            g.node_id = None
        snap.nodes = []
        planner = GPUGroupPlanner(snap)
        groups = planner.plan(2, 8)
        assert len(groups) == 8 and sorted(g for grp in groups for g in grp) == list(range(16))

    def test_insufficient_gpus(self) -> None:
        with pytest.raises(NoViablePlanError):
            GPUGroupPlanner(fh.h100x2()).plan(2, 2)

    def test_groups_never_span_nodes(self) -> None:
        planner = GPUGroupPlanner(fh.cluster(2, 4))
        groups = planner.plan(2, 4)
        assert groups == [[0, 1], [2, 3], [4, 5], [6, 7]]
        with pytest.raises(NoViablePlanError):
            planner.plan(8, 1)
        assert planner.plan(8, 1, allow_cross_node=True) == [list(range(8))]

    def test_describe(self) -> None:
        planner = GPUGroupPlanner(fh.nvlink_pairs_x4())
        desc = planner.describe([[0, 1], [0, 2], [3]])
        assert desc[0].endswith("(NVLink)") and "PCIe" in desc[1] and desc[2] == "GPU 3"
        assert (
            "topology unknown"
            in GPUGroupPlanner(fh.topology_unavailable_x4()).describe([[0, 1]])[0]
        )
