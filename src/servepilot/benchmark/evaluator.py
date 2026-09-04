"""Candidate evaluation: launch a candidate topology, benchmark it, tear it down.

:class:`LaunchingEvaluator` is the production implementation. It launches *all* replicas of the
candidate plus the same router/proxy used in production on an ephemeral port, so measurements
reflect exactly what users will get. Tests use scripted evaluators implementing the same
:class:`CandidateEvaluator` protocol.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, Protocol

from servepilot.api.app import ServingContext, create_app
from servepilot.benchmark.runner import BenchmarkRunner
from servepilot.constants import (
    DEFAULT_STARTUP_STAGGER_SECONDS,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    GIB,
)
from servepilot.engines.process import Launcher
from servepilot.engines.registry import EngineRegistry
from servepilot.hardware.base import HardwareProvider
from servepilot.logging import get_logger
from servepilot.models.tokenizer import TokenCounter
from servepilot.runtime.ports import PortAllocator, ephemeral_port
from servepilot.runtime.replicas import ReplicaLaunchError, ReplicaSet
from servepilot.runtime.router import ReplicaRouter
from servepilot.runtime.server import HTTPServer
from servepilot.schemas.benchmark import (
    BenchmarkResult,
    BenchmarkSpec,
    CandidateFailure,
    FailureType,
)
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidatePlan
from servepilot.schemas.workload import WorkloadProfile

log = get_logger(__name__)

# After a candidate is stopped, free GPU memory must return to within this much of the baseline.
CLEANUP_MEMORY_TOLERANCE_BYTES = 2 * GIB
CLEANUP_MEMORY_TIMEOUT_SECONDS = 60.0


class CandidateLaunchFailed(Exception):
    def __init__(self, failure: CandidateFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class CandidateSession(Protocol):
    plan: CandidatePlan
    launch_seconds: float
    engine_version: str | None
    runtime_metadata: dict[str, Any]

    async def benchmark(self, spec: BenchmarkSpec) -> BenchmarkResult: ...

    async def close(self) -> None: ...


class CandidateEvaluator(Protocol):
    async def open(self, plan: CandidatePlan) -> CandidateSession:
        """Launch ``plan``; raises :class:`CandidateLaunchFailed` with a classified failure."""
        ...


class LaunchedSession:
    def __init__(
        self,
        *,
        plan: CandidatePlan,
        replica_set: ReplicaSet,
        router: ReplicaRouter,
        server: HTTPServer,
        ctx: ServingContext,
        runner: BenchmarkRunner,
        served_model_name: str,
        launch_seconds: float,
        cleanup: Callable[[], Any],
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self.plan = plan
        self.replica_set = replica_set
        self.router = router
        self.server = server
        self.ctx = ctx
        self.runner = runner
        self.served_model_name = served_model_name
        self.launch_seconds = launch_seconds
        self.engine_version = replica_set.engine_version()
        self.runtime_metadata: dict[str, Any] = (
            dict(replica_set.replicas[0].metadata) if replica_set.replicas else {}
        )
        self._cleanup = cleanup
        self._progress = progress
        self._closed = False

    async def benchmark(self, spec: BenchmarkSpec) -> BenchmarkResult:
        return await self.runner.run(
            self.server.base_url,
            self.served_model_name,
            spec,
            candidate_id=self.plan.id,
            gpu_indices=self.plan.gpu_ids,
            progress=self._progress,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.server.stop()
        finally:
            ok = await self.replica_set.stop()
            await self.ctx.aclose()
            await self._cleanup()
            if not ok:
                log.error(
                    "candidate %s left processes behind; refusing to continue on contaminated hardware",
                    self.plan.id,
                )
                raise CandidateLaunchFailed(
                    CandidateFailure(
                        type=FailureType.ENGINE_CRASH,
                        message="engine processes could not be terminated after the benchmark",
                        stage="cleanup",
                    )
                )


class LaunchingEvaluator:
    def __init__(
        self,
        *,
        registry: EngineRegistry,
        launcher: Launcher,
        ports: PortAllocator,
        hardware_provider: HardwareProvider,
        hardware: HardwareSnapshot,
        model: ModelProfile,
        workload: WorkloadProfile,
        tokenizer: TokenCounter,
        served_model_name: str | None = None,
        trust_remote_code: bool = False,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        stagger_seconds: float = DEFAULT_STARTUP_STAGGER_SECONDS,
        progress: Callable[[int, int], None] | None = None,
        verify_gpu_cleanup: bool = True,
    ) -> None:
        self._registry = registry
        self._launcher = launcher
        self._ports = ports
        self._hw_provider = hardware_provider
        self._hardware = hardware
        self._model = model
        self._workload = workload
        self._tokenizer = tokenizer
        self._served = served_model_name or model.model_id
        self._trust = trust_remote_code
        self._timeout = startup_timeout
        self._stagger = stagger_seconds
        self._progress = progress
        self._verify_cleanup = verify_gpu_cleanup

    def _free_memory(self, gpu_ids: list[int]) -> dict[int, int]:
        snap = self._hw_provider.snapshot()
        out: dict[int, int] = {}
        for gid in gpu_ids:
            try:
                out[gid] = snap.gpu(gid).free_memory_bytes
            except KeyError:
                continue
        return out

    def revalidate_free_memory(self, plan: CandidatePlan) -> CandidateFailure | None:
        """Re-read free memory right before launch; fail fast if the plan can no longer fit."""
        est = plan.estimated_memory
        if est is None:
            return None
        free_now = self._free_memory(plan.gpu_ids)
        for gid, free in free_now.items():
            if free < est.engine_budget_bytes:
                return CandidateFailure(
                    type=FailureType.OOM,
                    message=(
                        f"GPU {gid} now has only {free / GIB:.1f} GiB free but the plan needs {est.engine_budget_bytes / GIB:.1f} GiB; "
                        "another process may be using the GPU"
                    ),
                    stage="prelaunch",
                )
        return None

    async def open(self, plan: CandidatePlan) -> CandidateSession:
        engine = self._registry.require(plan.engine)
        prelaunch = self.revalidate_free_memory(plan)
        if prelaunch is not None:
            raise CandidateLaunchFailed(prelaunch)
        baseline = self._free_memory(plan.gpu_ids) if self._verify_cleanup else {}

        router = ReplicaRouter(max_concurrency=None)
        replica_set = ReplicaSet(
            plan=plan,
            model=self._model,
            engine=engine,
            launcher=self._launcher,
            ports=self._ports,
            hardware=self._hardware,
            served_model_name=self._served,
            trust_remote_code=self._trust,
            startup_timeout=self._timeout,
            stagger_seconds=self._stagger,
            router=router,
        )
        start = time.monotonic()
        try:
            await replica_set.start()
        except ReplicaLaunchError as exc:
            await self._wait_for_cleanup(baseline)
            raise CandidateLaunchFailed(exc.failure) from exc
        launch_seconds = time.monotonic() - start

        ctx = ServingContext(
            router=router,
            model_id=self._model.model_id,
            served_model_name=self._served,
            backend_model_name=self._served,
            hardware=None,
            gpu_indices=plan.gpu_ids,
        )
        server = HTTPServer(create_app(ctx), "127.0.0.1", ephemeral_port())
        try:
            await server.start()
        except Exception:
            await replica_set.stop()
            await ctx.aclose()
            raise
        runner = BenchmarkRunner(
            tokenizer=self._tokenizer, workload=self._workload, hardware=self._hw_provider
        )

        async def cleanup() -> None:
            await self._wait_for_cleanup(baseline)

        return LaunchedSession(
            plan=plan,
            replica_set=replica_set,
            router=router,
            server=server,
            ctx=ctx,
            runner=runner,
            served_model_name=self._served,
            launch_seconds=launch_seconds,
            cleanup=cleanup,
            progress=self._progress,
        )

    async def _wait_for_cleanup(self, baseline: dict[int, int]) -> None:
        """Wait until GPU free memory is back near the pre-launch baseline (best effort, bounded)."""
        if not baseline or not self._verify_cleanup:
            return
        deadline = time.monotonic() + CLEANUP_MEMORY_TIMEOUT_SECONDS
        while True:
            try:
                now = await asyncio.to_thread(self._free_memory, list(baseline))
            except Exception as exc:
                log.debug("could not re-read GPU memory during cleanup: %s", exc)
                return
            short = {g: baseline[g] - now.get(g, baseline[g]) for g in baseline}
            worst = max(short.values(), default=0)
            if worst <= CLEANUP_MEMORY_TOLERANCE_BYTES:
                return
            if time.monotonic() > deadline:
                log.warning(
                    "GPU memory did not return to baseline after cleanup (%.1f GiB still in use on the worst GPU); "
                    "subsequent benchmarks may be affected",
                    worst / GIB,
                )
                return
            await asyncio.sleep(1.0)
