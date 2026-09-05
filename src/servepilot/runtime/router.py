"""Replica router: tracks backend replicas, applies the routing policy and backpressure."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from servepilot.constants import DEFAULT_MAX_QUEUE_DEPTH
from servepilot.runtime.policies import LeastInflightRoutingPolicy, RoutingPolicy
from servepilot.schemas.runtime import ReplicaState, ReplicaStatus


class NoHealthyReplicaError(Exception):
    """Raised when no replica can accept a request."""


class OverloadedError(Exception):
    """Raised when the wait queue is full."""


@dataclass
class RouteLease:
    """A request's claim on a replica; released via :meth:`ReplicaRouter.release`."""

    replica: ReplicaState
    acquired_at: float
    excluded: set[str]


class ReplicaRouter:
    def __init__(
        self,
        policy: RoutingPolicy | None = None,
        *,
        max_concurrency: int | None = None,
        max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH,
    ) -> None:
        self._policy = policy or LeastInflightRoutingPolicy()
        self._replicas: dict[str, ReplicaState] = {}
        self._max_concurrency = max_concurrency
        self._max_queue_depth = max_queue_depth
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrency) if max_concurrency else None
        )
        self._waiting = 0
        self.total_requests = 0
        self.total_errors = 0
        self.rejected_overload = 0

    # ------------------------------------------------------------------ replica management
    def add_replica(
        self,
        replica_id: str,
        base_url: str,
        *,
        gpu_ids: list[int] | None = None,
        pid: int | None = None,
    ) -> ReplicaState:
        state = ReplicaState(id=replica_id, base_url=base_url, gpu_ids=gpu_ids or [], pid=pid)
        self._replicas[replica_id] = state
        return state

    def remove_replica(self, replica_id: str) -> None:
        self._replicas.pop(replica_id, None)

    def set_status(self, replica_id: str, status: ReplicaStatus, error: str | None = None) -> None:
        r = self._replicas.get(replica_id)
        if r is None:
            return
        r.status = status
        if error is not None:
            r.last_error = error

    def replica(self, replica_id: str) -> ReplicaState | None:
        return self._replicas.get(replica_id)

    @property
    def replicas(self) -> list[ReplicaState]:
        return list(self._replicas.values())

    def healthy_replicas(self) -> list[ReplicaState]:
        return [r for r in self._replicas.values() if r.status == ReplicaStatus.HEALTHY]

    @property
    def max_concurrency(self) -> int | None:
        return self._max_concurrency

    def set_max_concurrency(self, value: int | None) -> None:
        self._max_concurrency = value
        self._semaphore = asyncio.Semaphore(value) if value else None

    @property
    def queue_depth(self) -> int:
        return self._waiting

    @property
    def inflight(self) -> int:
        return sum(r.inflight_requests for r in self._replicas.values())

    # ------------------------------------------------------------------ routing
    def select(
        self, excluded: set[str] | None = None, request_metadata: Mapping[str, Any] | None = None
    ) -> ReplicaState:
        candidates = [r for r in self.healthy_replicas() if not excluded or r.id not in excluded]
        if not candidates:
            raise NoHealthyReplicaError("no healthy replica available")
        return self._policy.select(candidates, request_metadata)

    async def acquire(self, request_metadata: Mapping[str, Any] | None = None) -> RouteLease:
        """Wait for a concurrency slot (bounded queue) and pick a replica."""
        if self._semaphore is not None:
            if self._semaphore.locked() and self._waiting >= self._max_queue_depth:
                self.rejected_overload += 1
                raise OverloadedError(f"request queue is full ({self._waiting} waiting)")
            self._waiting += 1
            try:
                await self._semaphore.acquire()
            finally:
                self._waiting -= 1
        try:
            replica = self.select(request_metadata=request_metadata)
        except NoHealthyReplicaError:
            if self._semaphore is not None:
                self._semaphore.release()
            raise
        replica.inflight_requests += 1
        replica.total_requests += 1
        self.total_requests += 1
        return RouteLease(replica=replica, acquired_at=time.monotonic(), excluded={replica.id})

    def reroute(self, lease: RouteLease, *, error: str | None = None) -> ReplicaState:
        """Move a lease to another healthy replica (used for pre-response retries).

        The abandoned replica is charged one failure and gives up the in-flight slot. When no
        other replica is available :class:`NoHealthyReplicaError` is raised and the lease is
        left untouched, so the caller's :meth:`release` still balances the books exactly once.
        """
        replica = self.select(excluded=lease.excluded)
        previous = lease.replica
        previous.inflight_requests = max(0, previous.inflight_requests - 1)
        previous.failures += 1
        if error is not None:
            previous.last_error = error
        replica.inflight_requests += 1
        replica.total_requests += 1
        lease.replica = replica
        lease.excluded.add(replica.id)
        return replica

    def release(self, lease: RouteLease, *, failed: bool = False, error: str | None = None) -> None:
        lease.replica.inflight_requests = max(0, lease.replica.inflight_requests - 1)
        if failed:
            lease.replica.failures += 1
            self.total_errors += 1
            if error is not None:
                lease.replica.last_error = error
        if self._semaphore is not None:
            self._semaphore.release()

    # ------------------------------------------------------------------ introspection
    def snapshot(self) -> dict[str, Any]:
        return {
            "replicas": [r.model_dump() for r in self._replicas.values()],
            "healthy_replicas": len(self.healthy_replicas()),
            "total_replicas": len(self._replicas),
            "inflight": self.inflight,
            "queue_depth": self.queue_depth,
            "max_concurrency": self._max_concurrency,
            "max_queue_depth": self._max_queue_depth,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "rejected_overload": self.rejected_overload,
        }
