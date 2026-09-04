"""Routing policies."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from servepilot.schemas.runtime import ReplicaState


class RoutingPolicy(ABC):
    @abstractmethod
    def select(
        self, replicas: Sequence[ReplicaState], request_metadata: Mapping[str, Any] | None = None
    ) -> ReplicaState:
        """Choose one *healthy* replica from ``replicas`` (never empty)."""


class LeastInflightRoutingPolicy(RoutingPolicy):
    """Prefer the replica with the fewest in-flight requests; round-robin among ties.

    Least-inflight adapts to variable request lengths, unlike blind round-robin.
    """

    def __init__(self) -> None:
        self._tick = itertools.count()

    def select(
        self, replicas: Sequence[ReplicaState], request_metadata: Mapping[str, Any] | None = None
    ) -> ReplicaState:
        if not replicas:
            raise ValueError("no replicas to select from")
        minimum = min(r.inflight_requests for r in replicas)
        ties = [r for r in replicas if r.inflight_requests == minimum]
        if len(ties) == 1:
            return ties[0]
        return ties[next(self._tick) % len(ties)]


class RoundRobinRoutingPolicy(RoutingPolicy):
    def __init__(self) -> None:
        self._tick = itertools.count()

    def select(
        self, replicas: Sequence[ReplicaState], request_metadata: Mapping[str, Any] | None = None
    ) -> ReplicaState:
        if not replicas:
            raise ValueError("no replicas to select from")
        return replicas[next(self._tick) % len(replicas)]


POLICIES: dict[str, type[RoutingPolicy]] = {
    "least_inflight": LeastInflightRoutingPolicy,
    "round_robin": RoundRobinRoutingPolicy,
}


def make_policy(name: str) -> RoutingPolicy:
    try:
        return POLICIES[name]()
    except KeyError as exc:
        raise ValueError(
            f"unknown routing policy {name!r}; choose from {sorted(POLICIES)}"
        ) from exc
