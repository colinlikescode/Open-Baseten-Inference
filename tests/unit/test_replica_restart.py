"""Replica restarts through the health checker: retried within the budget, then given up."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

import httpx
import pytest

from servepilot.engines.base import InferenceEngine, LaunchSpec, ReadinessResult, SupportResult
from servepilot.engines.process import ProcessHandle
from servepilot.exceptions import LaunchError
from servepilot.runtime.health import HealthChecker
from servepilot.runtime.ports import PortAllocator
from servepilot.runtime.replicas import ReplicaSet
from servepilot.runtime.router import ReplicaRouter
from servepilot.schemas.benchmark import CandidateFailure, FailureType
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidatePlan, EngineName
from servepilot.schemas.runtime import ReplicaStatus
from servepilot.testing import fake_hardware as fh


class _Process:
    """In-memory stand-in for a ManagedProcess."""

    def __init__(self, spec: LaunchSpec) -> None:
        self.spec = spec
        self.pid = 4242
        self.create_time = 0.0
        self.returncode: int | None = None

    def is_running(self) -> bool:
        return self.returncode is None

    def stdout_tail(self) -> str:
        return ""

    def stderr_tail(self) -> str:
        return ""

    async def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode

    async def terminate(self, grace_seconds: float = 20.0) -> None:
        if self.returncode is None:
            self.returncode = -15

    def die(self, code: int = 137) -> None:
        self.returncode = code


class _Launcher:
    def __init__(self) -> None:
        self.processes: list[_Process] = []
        self.fail_launches = 0

    async def launch(self, spec: LaunchSpec) -> ProcessHandle:
        if self.fail_launches > 0:
            self.fail_launches -= 1
            raise LaunchError("executable not found: stub")
        proc = _Process(spec)
        self.processes.append(proc)
        return proc

    async def shutdown_all(self, grace_seconds: float = 20.0) -> None:
        for proc in self.processes:
            await proc.terminate()

    def tracked(self) -> list[ProcessHandle]:
        return [p for p in self.processes if p.is_running()]

    def supports_node(self, node_id: str | None) -> bool:
        return True


class _Engine(InferenceEngine):
    """Readiness is scripted per launch: True = ready, False = fails to start."""

    engine_name = EngineName.FAKE

    def __init__(self, outcomes: list[bool]) -> None:
        self.outcomes = deque(outcomes)

    def name(self) -> str:
        return "stub"

    def is_available(self) -> bool:
        return True

    def version(self) -> str | None:
        return "0"

    def supports(self, model: ModelProfile, plan: CandidatePlan) -> SupportResult:
        return SupportResult(supported=True)

    def build_launch_spec(
        self,
        model: ModelProfile,
        plan: CandidatePlan,
        *,
        replica_index: int,
        host: str,
        port: int,
        served_model_name: str | None = None,
        trust_remote_code: bool = False,
        node: tuple[str, str] | None = None,
        local_gpu_ids: list[int] | None = None,
    ) -> LaunchSpec:
        return LaunchSpec(
            executable="stub",
            args=[],
            env={},
            host=host,
            port=port,
            gpu_ids=list(plan.gpu_groups[replica_index]),
            redacted_display_command="stub",
            replica_id=f"{plan.id}-r{replica_index}",
        )

    async def wait_until_ready(
        self,
        spec: LaunchSpec,
        process: ProcessHandle | None,
        timeout_seconds: float,
        poll_interval: float = 2.0,
    ) -> ReadinessResult:
        ready = self.outcomes.popleft() if self.outcomes else True
        if ready:
            return ReadinessResult(True, 0.1)
        failure = CandidateFailure(
            type=FailureType.OOM, message="CUDA out of memory", stage="startup"
        )
        return ReadinessResult(False, 0.1, failure)


async def _probe_ok(self: HealthChecker, client: httpx.AsyncClient, replica: Any) -> bool:
    return True


async def _start(engine: _Engine, launcher: _Launcher, router: ReplicaRouter) -> ReplicaSet:
    plan = CandidatePlan(
        id="stub-tp1-x2",
        engine=EngineName.FAKE,
        gpu_groups=[[0], [1]],
        tensor_parallel_size=1,
        replica_count=2,
        context_length=1024,
    )
    replica_set = ReplicaSet(
        plan=plan,
        model=ModelProfile(model_id="m"),
        engine=engine,
        launcher=launcher,
        ports=PortAllocator(37000, 37999),
        hardware=fh.h100x2(),
        router=router,
        stagger_seconds=0,
    )
    await replica_set.start()
    assert len(router.healthy_replicas()) == 2
    return replica_set


def _checker(router: ReplicaRouter, replica_set: ReplicaSet, max_restarts: int) -> HealthChecker:
    return HealthChecker(
        router,
        replica_set,
        interval=0.01,
        unhealthy_after=3,
        max_restarts=max_restarts,
        restart_window=600.0,
        restart_backoff=0.0,
    )


async def _settle(checker: HealthChecker) -> None:
    for _ in range(500):
        if not checker._restarting:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("restart task did not finish")


async def test_failed_restart_is_retried_and_counts_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(HealthChecker, "_probe", _probe_ok)
    # Two initial launches succeed; the first restart fails at readiness, the second succeeds.
    engine = _Engine([True, True, False, True])
    launcher = _Launcher()
    router = ReplicaRouter()
    replica_set = await _start(engine, launcher, router)
    checker = _checker(router, replica_set, max_restarts=3)
    dead = replica_set.replicas[0]
    assert isinstance(dead.process, _Process)
    dead.process.die()

    async with httpx.AsyncClient() as client:
        await checker.check_once(client)
        await _settle(checker)
        state = router.replica(dead.id)
        assert state is not None and state.status == ReplicaStatus.UNHEALTHY
        assert state.restarts == 1 and "CUDA out of memory" in (state.last_error or "")
        assert replica_set.replicas[0] is dead, "the slot must stay visible for another attempt"
        assert any("restart failed" in e for e in checker.events)
        assert len(router.healthy_replicas()) == 1

        await checker.check_once(client)
        await _settle(checker)
        fresh = replica_set.replicas[0]
        assert fresh is not dead and fresh.ready and fresh.index == 0
        assert fresh.spec.port == dead.spec.port
        state = router.replica(fresh.id)
        assert state is not None and state.status == ReplicaStatus.HEALTHY
        assert state.restarts == 2
        assert len(router.healthy_replicas()) == 2
        assert any("restarted successfully" in e for e in checker.events)
    await replica_set.stop()


async def test_stop_abandons_pending_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HealthChecker, "_probe", _probe_ok)
    engine = _Engine([True, True])
    launcher = _Launcher()
    router = ReplicaRouter()
    replica_set = await _start(engine, launcher, router)
    checker = HealthChecker(
        router, replica_set, interval=0.01, max_restarts=3, restart_backoff=60.0
    )  # a long backoff keeps the restart pending
    dead = replica_set.replicas[0]
    assert isinstance(dead.process, _Process)
    dead.process.die()
    async with httpx.AsyncClient() as client:
        await checker.check_once(client)
    await asyncio.sleep(0)  # let the restart task start and park in its backoff sleep
    assert checker._restarting == {0} and len(checker._restart_tasks) == 1
    assert any("restarting" in e for e in checker.events)
    launched = len(launcher.processes)
    await checker.stop()
    assert not checker._restart_tasks and not checker._restarting
    assert len(launcher.processes) == launched, "no engine may be launched after stop()"
    await replica_set.stop()


async def test_restart_budget_exhaustion_stops_replica(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HealthChecker, "_probe", _probe_ok)
    engine = _Engine([True, True])
    launcher = _Launcher()
    router = ReplicaRouter()
    replica_set = await _start(engine, launcher, router)
    checker = _checker(router, replica_set, max_restarts=2)
    launcher.fail_launches = 2  # both restart attempts fail before a process exists
    dead = replica_set.replicas[1]
    assert isinstance(dead.process, _Process)
    dead.process.die()

    async with httpx.AsyncClient() as client:
        for attempt in (1, 2):
            await checker.check_once(client)
            await _settle(checker)
            state = router.replica(dead.id)
            assert state is not None and state.status == ReplicaStatus.UNHEALTHY
            assert state.restarts == attempt and "executable not found" in (state.last_error or "")
        await checker.check_once(client)
        await _settle(checker)
        state = router.replica(dead.id)
        assert state is not None and state.status == ReplicaStatus.STOPPED
        assert any("exceeded 2 restarts" in e for e in checker.events)
        launched_before = len(launcher.processes)
        await checker.check_once(client)
        await _settle(checker)
        assert len(launcher.processes) == launched_before, "a STOPPED replica is never relaunched"
        assert [r.id for r in router.healthy_replicas()] == [replica_set.replicas[0].id]
    await replica_set.stop()
