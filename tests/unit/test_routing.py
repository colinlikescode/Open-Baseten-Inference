"""Routing policies, router bookkeeping and backpressure."""

from __future__ import annotations

import asyncio

import pytest

from servepilot.runtime.policies import (
    LeastInflightRoutingPolicy,
    RoundRobinRoutingPolicy,
    make_policy,
)
from servepilot.runtime.router import NoHealthyReplicaError, OverloadedError, ReplicaRouter
from servepilot.schemas.runtime import ReplicaState, ReplicaStatus


def replica(rid: str, inflight: int = 0, healthy: bool = True) -> ReplicaState:
    return ReplicaState(
        id=rid,
        base_url=f"http://{rid}",
        inflight_requests=inflight,
        status=ReplicaStatus.HEALTHY if healthy else ReplicaStatus.UNHEALTHY,
    )


class TestPolicies:
    def test_least_inflight(self) -> None:
        policy = LeastInflightRoutingPolicy()
        a, b, c = replica("a", 3), replica("b", 1), replica("c", 2)
        assert policy.select([a, b, c]).id == "b"

    def test_ties_round_robin(self) -> None:
        policy = LeastInflightRoutingPolicy()
        a, b = replica("a", 0), replica("b", 0)
        picks = [policy.select([a, b]).id for _ in range(4)]
        assert picks == ["a", "b", "a", "b"]

    def test_round_robin(self) -> None:
        policy = RoundRobinRoutingPolicy()
        reps = [replica("a"), replica("b"), replica("c")]
        assert [policy.select(reps).id for _ in range(4)] == ["a", "b", "c", "a"]

    def test_empty_and_factory(self) -> None:
        with pytest.raises(ValueError, match="no replicas"):
            LeastInflightRoutingPolicy().select([])
        assert isinstance(make_policy("round_robin"), RoundRobinRoutingPolicy)
        with pytest.raises(ValueError, match="unknown routing policy"):
            make_policy("random")


class TestRouter:
    async def test_unhealthy_excluded_and_bookkeeping(self) -> None:
        router = ReplicaRouter()
        router.add_replica("a", "http://a")
        router.add_replica("b", "http://b")
        router.set_status("a", ReplicaStatus.HEALTHY)
        router.set_status("b", ReplicaStatus.UNHEALTHY, "probe failed")
        lease = await router.acquire()
        assert lease.replica.id == "a" and lease.replica.inflight_requests == 1
        router.release(lease)
        assert lease.replica.inflight_requests == 0 and router.total_requests == 1
        router.set_status("a", ReplicaStatus.STOPPED)
        with pytest.raises(NoHealthyReplicaError):
            await router.acquire()
        snap = router.snapshot()
        assert snap["healthy_replicas"] == 0 and snap["total_replicas"] == 2

    async def test_reroute_excludes_previous(self) -> None:
        router = ReplicaRouter()
        for rid in ("a", "b"):
            router.add_replica(rid, f"http://{rid}")
            router.set_status(rid, ReplicaStatus.HEALTHY)
        lease = await router.acquire()
        first = lease.replica.id
        other = router.reroute(lease, error="ConnectError: refused")
        assert other.id != first and lease.replica is other
        abandoned = router.replica(first)
        assert abandoned is not None and abandoned.failures == 1
        assert abandoned.inflight_requests == 0 and abandoned.last_error == "ConnectError: refused"
        assert other.inflight_requests == 1
        with pytest.raises(NoHealthyReplicaError):
            router.reroute(lease)  # nothing left to try
        # A failed reroute leaves the lease untouched: the slot is still on `other`, once.
        assert lease.replica is other and other.inflight_requests == 1
        router.release(lease, failed=True, error="HTTP 500")
        assert router.total_errors == 1 and other.inflight_requests == 0
        assert other.failures == 1 and other.last_error == "HTTP 500"
        assert abandoned.inflight_requests == 0, "the abandoned replica must not be charged twice"

    async def test_backpressure_queue_and_overload(self) -> None:
        router = ReplicaRouter(max_concurrency=1, max_queue_depth=1)
        router.add_replica("a", "http://a")
        router.set_status("a", ReplicaStatus.HEALTHY)
        first = await router.acquire()
        waiter = asyncio.create_task(router.acquire())
        await asyncio.sleep(0.01)
        assert router.queue_depth == 1
        with pytest.raises(OverloadedError):
            await router.acquire()
        assert router.rejected_overload == 1
        router.release(first)
        second = await waiter
        assert second.replica.id == "a" and router.queue_depth == 0
        router.release(second)
        router.set_max_concurrency(None)
        assert router.max_concurrency is None
