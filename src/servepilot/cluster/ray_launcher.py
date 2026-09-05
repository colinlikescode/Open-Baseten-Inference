"""Launch engine processes on Ray worker nodes.

An :class:`EngineProcessActor` (one per engine process) runs on the target node and owns a local
:class:`ManagedProcess`; :class:`RayProcessHandle` mirrors its state to the driver so the tuner,
health checker and readiness probes use the same :class:`ProcessHandle` interface everywhere.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from servepilot.cluster import require_ray
from servepilot.constants import DEFAULT_GRACEFUL_SHUTDOWN_SECONDS
from servepilot.engines.base import LaunchSpec
from servepilot.engines.process import ManagedProcess, ProcessHandle
from servepilot.exceptions import LaunchError
from servepilot.logging import get_logger

log = get_logger(__name__)

STATUS_POLL_INTERVAL_SECONDS = 1.0


class EngineProcessActor:
    """Ray actor body (decorated lazily so importing this module never needs Ray)."""

    def __init__(self) -> None:
        self._proc: ManagedProcess | None = None

    async def start(self, spec_json: str) -> dict[str, Any]:
        spec = LaunchSpec.model_validate_json(spec_json)
        self._proc = ManagedProcess(spec)
        await self._proc.start()
        return {"pid": self._proc.pid, "create_time": self._proc.create_time}

    def status(self) -> dict[str, Any]:
        assert self._proc is not None
        return {
            "running": self._proc.is_running(),
            "returncode": self._proc.returncode,
            "stdout": self._proc.stdout_tail(),
            "stderr": self._proc.stderr_tail(),
        }

    async def wait(self, timeout: float | None) -> int | None:
        assert self._proc is not None
        return await self._proc.wait(timeout)

    async def terminate(self, grace_seconds: float) -> None:
        if self._proc is not None:
            await self._proc.terminate(grace_seconds)

    def ping(self) -> bool:
        return True


class RayProcessHandle:
    """Driver-side view of a remote engine process."""

    def __init__(self, spec: LaunchSpec, actor: Any) -> None:
        self.spec = spec
        self._actor = actor
        self._pid: int | None = None
        self._create_time: float | None = None
        self._running = True
        self._returncode: int | None = None
        self._stdout = ""
        self._stderr = ""
        self._monitor: asyncio.Task[None] | None = None

    async def start(self) -> None:
        info = await _await_ref(self._actor.start.remote(self.spec.model_dump_json()))
        self._pid = info.get("pid")
        self._create_time = info.get("create_time")
        await self.refresh()
        self._monitor = asyncio.create_task(self._poll())

    async def refresh(self) -> None:
        status = await _await_ref(self._actor.status.remote())
        self._running = bool(status["running"])
        self._returncode = status["returncode"]
        self._stdout = status["stdout"]
        self._stderr = status["stderr"]

    async def _poll(self) -> None:
        while True:
            await asyncio.sleep(STATUS_POLL_INTERVAL_SECONDS)
            try:
                await self.refresh()
            except Exception as exc:
                log.debug("status poll for %s failed: %s", self.spec.replica_id, exc)
                self._running = False
                return
            if not self._running:
                return

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def create_time(self) -> float | None:
        return self._create_time

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def is_running(self) -> bool:
        return self._running

    def stdout_tail(self) -> str:
        return self._stdout

    def stderr_tail(self) -> str:
        return self._stderr

    async def wait(self, timeout: float | None = None) -> int | None:
        try:
            code = await asyncio.wait_for(
                _await_ref(self._actor.wait.remote(timeout)),
                timeout=(timeout or 0) + 30 if timeout else None,
            )
        except TimeoutError:
            return None
        await self.refresh()
        return code  # type: ignore[no-any-return]

    async def terminate(self, grace_seconds: float = DEFAULT_GRACEFUL_SHUTDOWN_SECONDS) -> None:
        if self._monitor is not None:
            self._monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor
            self._monitor = None
        try:
            await _await_ref(self._actor.terminate.remote(grace_seconds))
            await self.refresh()
        except Exception as exc:
            log.debug("terminate on %s: %s", self.spec.replica_id, exc)
        self._running = False
        ray = require_ray()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(ray.kill, self._actor, no_restart=True)  # synchronous RPC


async def _await_ref(ref: Any) -> Any:
    return await asyncio.wrap_future(ref.future())


class RayLauncher:
    """A :class:`Launcher` placing each engine process on the node named in its spec."""

    def __init__(self, address: str = "auto") -> None:
        self.address = address
        self._handles: list[RayProcessHandle] = []
        self._actor_cls: Any | None = None

    def _cls(self) -> Any:
        ray = require_ray()
        if not ray.is_initialized():
            ray.init(address=self.address, ignore_reinit_error=True, log_to_driver=False)
        if self._actor_cls is None:
            self._actor_cls = ray.remote(num_cpus=0)(EngineProcessActor)
        return self._actor_cls

    async def launch(self, spec: LaunchSpec) -> ProcessHandle:
        def create_actor() -> Any:
            # Connecting to Ray (on first use) and creating the actor are synchronous RPCs.
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            cls = self._cls()
            options: dict[str, Any] = {"name": None}
            if spec.node_id:
                options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                    node_id=spec.node_id, soft=False
                )
            return cls.options(**options).remote()

        try:
            actor = await asyncio.to_thread(create_actor)
            handle = RayProcessHandle(spec, actor)
            await handle.start()
        except Exception as exc:
            raise LaunchError(
                f"could not start {spec.replica_id} on Ray node {spec.node_id or 'any'}: {exc}",
                hints=["Confirm the node is alive in `ray status` and has ServePilot installed."],
            ) from exc
        self._handles.append(handle)
        return handle

    async def shutdown_all(self, grace_seconds: float = DEFAULT_GRACEFUL_SHUTDOWN_SECONDS) -> None:
        handles, self._handles = self._handles, []
        await asyncio.gather(*(h.terminate(grace_seconds) for h in handles), return_exceptions=True)

    def tracked(self) -> list[ProcessHandle]:
        return [h for h in self._handles if h.is_running()]

    def supports_node(self, node_id: str | None) -> bool:
        return True
