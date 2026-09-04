"""Assign GPUs to tensor-parallel groups using interconnect topology.

The planner maximises intra-group connectivity (sum of pairwise :meth:`TopologyEdge.score`),
strongly preferring NVLink, then closer PCIe ancestry. Groups never span machines unless the
caller explicitly allows it (cross-node replicas use an engine's distributed executor).

For small machines (≤ 8 GPUs) the partition space is enumerated exhaustively; larger machines
use a deterministic greedy heuristic. Ties are broken by lowest GPU indices so results are stable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

from servepilot.exceptions import NoViablePlanError
from servepilot.schemas.hardware import HardwareSnapshot

EXHAUSTIVE_MAX_GPUS = 8


class GPUGroupPlanner:
    def __init__(self, snapshot: HardwareSnapshot) -> None:
        self._snapshot = snapshot
        self._node_of = {g.index: g.node_id for g in snapshot.gpus}
        self._score_cache: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------ scoring
    def pair_score(self, a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in self._score_cache:
            if self._node_of.get(a) != self._node_of.get(b):
                score = 0
            elif not self._snapshot.topology.available:
                # Without topology, prefer adjacent indices: many systems wire consecutive GPUs
                # to the same PCIe switch, and it keeps results deterministic.
                score = max(1, 8 - abs(a - b))
            else:
                score = self._snapshot.topology.score(a, b)
            self._score_cache[key] = score
        return self._score_cache[key]

    def group_score(self, group: Sequence[int]) -> int:
        return sum(self.pair_score(a, b) for a, b in combinations(sorted(group), 2))

    def partition_score(self, groups: Iterable[Sequence[int]]) -> int:
        return sum(self.group_score(g) for g in groups)

    # ------------------------------------------------------------------ planning
    def plan(
        self,
        tp_size: int,
        replica_count: int,
        gpu_ids: Sequence[int] | None = None,
        *,
        allow_cross_node: bool = False,
    ) -> list[list[int]]:
        """Return ``replica_count`` groups of ``tp_size`` GPUs each."""
        ids = sorted(gpu_ids if gpu_ids is not None else [g.index for g in self._snapshot.gpus])
        needed = tp_size * replica_count
        if needed > len(ids):
            raise NoViablePlanError(
                f"TP={tp_size} × {replica_count} replicas needs {needed} GPUs but only {len(ids)} are selected"
            )
        if tp_size == 1:
            return [[i] for i in ids[:replica_count]]

        if not allow_cross_node and len({self._node_of[i] for i in ids}) > 1:
            return self._plan_per_node(tp_size, replica_count, ids)

        if len(ids) <= EXHAUSTIVE_MAX_GPUS:
            groups = self._exhaustive(tp_size, replica_count, ids)
        else:
            groups = self._greedy(tp_size, replica_count, ids)
        return [sorted(g) for g in sorted(groups, key=lambda g: min(g))]

    def _plan_per_node(self, tp_size: int, replica_count: int, ids: list[int]) -> list[list[int]]:
        by_node: dict[str | None, list[int]] = {}
        for i in ids:
            by_node.setdefault(self._node_of[i], []).append(i)
        groups: list[list[int]] = []
        for node in sorted(by_node, key=lambda n: (n is None, str(n))):
            node_ids = by_node[node]
            per_node = len(node_ids) // tp_size
            if per_node == 0:
                continue
            groups.extend(self.plan(tp_size, per_node, node_ids, allow_cross_node=True))
        if len(groups) < replica_count:
            raise NoViablePlanError(
                f"TP={tp_size} groups cannot span machines; only {len(groups)} group(s) fit within nodes "
                f"but {replica_count} replicas were requested"
            )
        return groups[:replica_count]

    def _exhaustive(self, tp_size: int, replica_count: int, ids: list[int]) -> list[list[int]]:
        best: list[list[int]] | None = None
        best_score = -1

        def recurse(remaining: list[int], chosen: list[list[int]]) -> None:
            nonlocal best, best_score
            if len(chosen) == replica_count:
                score = self.partition_score(chosen)
                if score > best_score:
                    best_score = score
                    best = [list(g) for g in chosen]
                return
            # Anchor on the smallest remaining id to avoid enumerating permutations of the
            # same partition; when GPUs are left over, also allow skipping the anchor.
            if len(remaining) < tp_size * (replica_count - len(chosen)):
                return
            anchor, rest = remaining[0], remaining[1:]
            for combo in combinations(rest, tp_size - 1):
                group = [anchor, *combo]
                left = [r for r in rest if r not in combo]
                recurse(left, [*chosen, group])
            if len(remaining) > tp_size * (replica_count - len(chosen)):
                recurse(rest, chosen)

        recurse(ids, [])
        if best is None:  # pragma: no cover - guarded by the size check in plan()
            raise NoViablePlanError("could not partition GPUs into tensor-parallel groups")
        return best

    def _greedy(self, tp_size: int, replica_count: int, ids: list[int]) -> list[list[int]]:
        remaining = list(ids)
        groups: list[list[int]] = []
        while len(groups) < replica_count:
            # Seed with the best-connected remaining GPU (ties → lowest index).
            seed = max(
                remaining,
                key=lambda i: (sum(self.pair_score(i, j) for j in remaining if j != i), -i),
            )
            group = [seed]
            remaining.remove(seed)
            while len(group) < tp_size:
                nxt = max(remaining, key=lambda c: (sum(self.pair_score(c, m) for m in group), -c))
                group.append(nxt)
                remaining.remove(nxt)
            groups.append(sorted(group))
        return groups

    def describe(self, groups: Sequence[Sequence[int]]) -> list[str]:
        """Human-readable connectivity description for each group."""
        out: list[str] = []
        for g in groups:
            if len(g) == 1:
                out.append(f"GPU {g[0]}")
                continue
            pairs = list(combinations(sorted(g), 2))
            nv = sum(
                1
                for a, b in pairs
                if (e := self._snapshot.topology.edge(a, b)) and e.nvlink_detected
            )
            if not self._snapshot.topology.available:
                conn = "topology unknown"
            elif nv == len(pairs):
                conn = "NVLink"
            elif nv:
                conn = f"partial NVLink ({nv}/{len(pairs)} pairs)"
            else:
                rels: set[str] = set()
                for a, b in pairs:
                    edge = self._snapshot.topology.edge(a, b)
                    rels.add(edge.relationship if edge is not None else "UNKNOWN")
                conn = "PCIe " + "/".join(sorted(rels))
            out.append(f"GPUs {list(g)} ({conn})")
        return out
