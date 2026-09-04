"""GPU topology helpers.

The NVML-specific probing lives in :mod:`servepilot.hardware.nvml`; this module holds the pure
functions that turn raw NVML answers into a :class:`GPUTopology`, so they can be unit tested
without a GPU.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from servepilot.schemas.hardware import GPUTopology, TopologyEdge, TopologyRelationship

# NVML ``nvmlDeviceGetTopologyCommonAncestor`` level codes.
NVML_TOPOLOGY_INTERNAL = 0
NVML_TOPOLOGY_SINGLE = 10
NVML_TOPOLOGY_MULTIPLE = 20
NVML_TOPOLOGY_HOSTBRIDGE = 30
NVML_TOPOLOGY_NODE = 40
NVML_TOPOLOGY_SYSTEM = 50

_ANCESTOR_TO_RELATIONSHIP: dict[int, TopologyRelationship] = {
    NVML_TOPOLOGY_INTERNAL: TopologyRelationship.PIX,
    NVML_TOPOLOGY_SINGLE: TopologyRelationship.PIX,
    NVML_TOPOLOGY_MULTIPLE: TopologyRelationship.PXB,
    NVML_TOPOLOGY_HOSTBRIDGE: TopologyRelationship.PHB,
    NVML_TOPOLOGY_NODE: TopologyRelationship.NODE,
    NVML_TOPOLOGY_SYSTEM: TopologyRelationship.SYS,
}


def ancestor_to_relationship(level: int | None) -> TopologyRelationship:
    """Map an NVML common-ancestor level to a :class:`TopologyRelationship`."""
    if level is None:
        return TopologyRelationship.UNKNOWN
    return _ANCESTOR_TO_RELATIONSHIP.get(level, TopologyRelationship.UNKNOWN)


def build_topology(
    gpu_indices: Iterable[int],
    pcie_relationships: Mapping[tuple[int, int], TopologyRelationship | str],
    nvlink_counts: Mapping[tuple[int, int], int],
    *,
    nvswitch_links: Mapping[int, int] | None = None,
) -> GPUTopology:
    """Assemble a :class:`GPUTopology` from pairwise facts.

    ``pcie_relationships`` and ``nvlink_counts`` are keyed by ``(a, b)`` with ``a < b``.
    ``nvswitch_links`` maps a GPU index to the number of NVLink links that terminate at an
    NVSwitch; when every GPU in a pair has NVSwitch links they are treated as fully connected.
    """
    indices = sorted(set(gpu_indices))
    edges: list[TopologyEdge] = []
    for i, a in enumerate(indices):
        for b in indices[i + 1 :]:
            key = (a, b)
            rel = pcie_relationships.get(key, TopologyRelationship.UNKNOWN)
            rel_value = rel.value if isinstance(rel, TopologyRelationship) else str(rel)
            links = nvlink_counts.get(key, 0)
            if nvswitch_links and nvswitch_links.get(a, 0) > 0 and nvswitch_links.get(b, 0) > 0:
                links = max(links, min(nvswitch_links[a], nvswitch_links[b]))
            nvlink = links > 0
            edges.append(
                TopologyEdge(
                    gpu_a=a,
                    gpu_b=b,
                    relationship=TopologyRelationship.NVLINK.value if nvlink else rel_value,
                    nvlink_detected=nvlink,
                    nvlink_link_count=links if nvlink else None,
                )
            )
    return GPUTopology(edges=edges, available=True)


def unavailable_topology(error: str) -> GPUTopology:
    """Topology placeholder when NVML could not answer; planners fall back to index grouping."""
    return GPUTopology(edges=[], available=False, error=error)
