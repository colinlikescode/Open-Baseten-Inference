"""Subprocess supervision for engine servers.

Every engine process is started in its own session/process group so that the whole tree
(engine + worker processes it forks) can be terminated with one ``killpg``. Output is captured
into bounded ring buffers for failure classification and verbose streaming. A module-level
registry plus an ``atexit`` hook guarantees that no engine process outlives ServePilot even if
the interpreter exits unexpectedly.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import signal
import time
from collections import deque
from collections.abc import Callable
from typing import Protocol

import psutil

from servepilot.constants import (
    DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
    PROCESS_LOG_TAIL_BYTES,
    PROCESS_LOG_TAIL_LINES,
)
from servepilot.engines.base import LaunchSpec
from servepilot.exceptions import LaunchError
from servepilot.logging import get_logger, redact_secrets

log = get_logger(__name__)

LineCallback = Callable[[str, str], None]  # (stream, line)


class RingBuffer:
    """Bounded line buffer (by line count and total bytes)."""

    def __init__(
        self, max_lines: int = PROCESS_LOG_TAIL_LINES, max_bytes: int = PROCESS_LOG_TAIL_BYTES
    ) -> None:
        self._lines: deque[str] = deque()
        self._bytes = 0
        self._max_lines = max_lines
        self._max_bytes = max_bytes

    def append(self, line: str) -> None:
        self._lines.append(line)
        self._bytes += len(line)
        while len(self._lines) > self._max_lines or (
            self._bytes > self._max_bytes and len(self._lines) > 1
        ):
            self._bytes -= len(self._lines.popleft())

    def text(self) -> str:
        return "\n".join(self._lines)

    def __len__(self) -> int:
        return len(self._lines)


class ProcessHandle(Protocol):
    """What the tuner/runtime need to know about a launched engine process."""

    spec: LaunchSpec

    @property
    def pid(self) -> int | None: ...

    @property
    def create_time(self) -> float | None: ...

    @property
    def returncode(self) -> int | None: ...

    def is_running(self) -> bool: ...

    def stdout_tail(self) -> str: ...

    def stderr_tail(self) -> str: ...

    async def wait(self, timeout: float | None = None) -> int | None: ...

    async def terminate(self, grace_seconds: float = DEFAULT_GRACEFUL_SHUTDOWN_SECONDS) -> None: ...


# --------------------------------------------------------------------------- emergency cleanup
_LIVE_PGIDS: set[int] = set()


def _emergency_cleanup() -> None:
    """Synchronous last-resort cleanup at interpreter exit."""
    for pgid in list(_LIVE_PGIDS):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while _LIVE_PGIDS and time.monotonic() < deadline:
        for pgid in list(_LIVE_PGIDS):
            if not _pgid_alive(pgid):
                _LIVE_PGIDS.discard(pgid)
        if _LIVE_PGIDS:
            time.sleep(0.1)
    for pgid in list(_LIVE_PGIDS):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)
        _LIVE_PGIDS.discard(pgid)


atexit.register(_emergency_cleanup)


def _pgid_alive(pgid: int) -> bool:
    for proc in psutil.process_iter(["pid"]):
        try:
            if os.getpgid(proc.info["pid"]) == pgid:
                return True
        except (ProcessLookupError, psutil.Error, PermissionError):
            continue
    return False


def _descendant_pids(pgid: int) -> list[int]:
    pids: list[int] = []
    for proc in psutil.process_iter(["pid"]):
        try:
            if os.getpgid(proc.info["pid"]) == pgid:
                pids.append(proc.info["pid"])
        except (ProcessLookupError, psutil.Error, PermissionError):
            continue
    return pids


# --------------------------------------------------------------------------- local process
class ManagedProcess:
    """An engine server started locally in its own process group."""

    def __init__(self, spec: LaunchSpec, on_line: LineCallback | None = None) -> None:
        self.spec = spec
        self._proc: asyncio.subprocess.Process | None = None
        self._stdout = RingBuffer()
        self._stderr = RingBuffer()
        self._readers: list[asyncio.Task[None]] = []
        self._on_line = on_line
        self._pgid: int | None = None
        self._create_time: float | None = None
        self.started_at: float | None = None

    # -- properties -------------------------------------------------------
    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def create_time(self) -> float | None:
        return self._create_time

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode if self._proc is not None else None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def stdout_tail(self) -> str:
        return self._stdout.text()

    def stderr_tail(self) -> str:
        return self._stderr.text()

    # -- lifecycle --------------------------------------------------------
    async def start(self) -> None:
        if self._proc is not None:
            raise LaunchError("process already started")
        log.info("launching %s", self.spec.redacted_display_command)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.spec.executable,
                *self.spec.args,
                env=self.spec.env,
                cwd=self.spec.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise LaunchError(
                f"executable not found: {self.spec.executable}",
                hints=[
                    f"Install the engine or set the interpreter path for {self.spec.replica_id}."
                ],
            ) from exc
        except OSError as exc:
            raise LaunchError(f"could not start {self.spec.executable}: {exc}") from exc
        self.started_at = time.time()
        try:
            self._pgid = os.getpgid(self._proc.pid)
            _LIVE_PGIDS.add(self._pgid)
        except ProcessLookupError:
            self._pgid = None
        with contextlib.suppress(psutil.Error):
            self._create_time = psutil.Process(self._proc.pid).create_time()
        assert self._proc.stdout is not None and self._proc.stderr is not None
        self._readers = [
            asyncio.create_task(self._pump(self._proc.stdout, self._stdout, "stdout")),
            asyncio.create_task(self._pump(self._proc.stderr, self._stderr, "stderr")),
        ]

    async def _pump(self, stream: asyncio.StreamReader, buffer: RingBuffer, name: str) -> None:
        while True:
            try:
                raw = await stream.readline()
            except (asyncio.LimitOverrunError, ValueError):
                # Extremely long line (progress bars); drain in chunks.
                raw = await stream.read(64 * 1024)
            if not raw:
                break
            line = redact_secrets(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
            buffer.append(line)
            if self._on_line is not None:
                self._on_line(name, line)

    async def wait(self, timeout: float | None = None) -> int | None:
        if self._proc is None:
            return None
        try:
            return await asyncio.wait_for(self._proc.wait(), timeout=timeout)
        except TimeoutError:
            return None

    def _signal_group(self, sig: signal.Signals) -> None:
        if self._pgid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self._pgid, sig)
        elif self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.send_signal(sig)

    async def terminate(self, grace_seconds: float = DEFAULT_GRACEFUL_SHUTDOWN_SECONDS) -> None:
        """SIGTERM the process group, wait, then SIGKILL anything left."""
        if self._proc is None:
            return
        if self._proc.returncode is None:
            log.debug("terminating pid %s (pgid %s)", self.pid, self._pgid)
            self._signal_group(signal.SIGTERM)
            code = await self.wait(timeout=grace_seconds)
            if code is None:
                log.warning(
                    "pid %s did not exit within %.0fs; sending SIGKILL", self.pid, grace_seconds
                )
                self._signal_group(signal.SIGKILL)
                await self.wait(timeout=10.0)
        # Workers may have re-parented; make sure nothing in the group survives. Scanning the
        # process table is synchronous and can take a while on a busy host, so do it off-loop.
        if self._pgid is not None:
            for pid in await asyncio.to_thread(_descendant_pids, self._pgid):
                with contextlib.suppress(ProcessLookupError, psutil.Error, PermissionError):
                    psutil.Process(pid).kill()
            _LIVE_PGIDS.discard(self._pgid)
        for task in self._readers:
            if not task.done():
                task.cancel()
        for task in self._readers:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._readers = []

    def group_alive(self) -> bool:
        return self._pgid is not None and _pgid_alive(self._pgid)


# --------------------------------------------------------------------------- launcher
class Launcher(Protocol):
    """Starts engine processes locally or on remote nodes (Ray)."""

    async def launch(self, spec: LaunchSpec) -> ProcessHandle: ...

    async def shutdown_all(
        self, grace_seconds: float = DEFAULT_GRACEFUL_SHUTDOWN_SECONDS
    ) -> None: ...

    def tracked(self) -> list[ProcessHandle]: ...

    def supports_node(self, node_id: str | None) -> bool: ...


class LocalLauncher:
    """Launches processes on this machine."""

    def __init__(self, on_line: LineCallback | None = None) -> None:
        self._on_line = on_line
        self._processes: list[ManagedProcess] = []

    async def launch(self, spec: LaunchSpec) -> ProcessHandle:
        proc = ManagedProcess(spec, on_line=self._on_line)
        await proc.start()
        self._processes.append(proc)
        return proc

    async def shutdown_all(self, grace_seconds: float = DEFAULT_GRACEFUL_SHUTDOWN_SECONDS) -> None:
        procs, self._processes = self._processes, []
        await asyncio.gather(*(p.terminate(grace_seconds) for p in procs), return_exceptions=True)

    def tracked(self) -> list[ProcessHandle]:
        return [p for p in self._processes if p.is_running()]

    def supports_node(self, node_id: str | None) -> bool:
        return node_id is None or node_id == "local"

    def forget(self, handle: ProcessHandle) -> None:
        self._processes = [p for p in self._processes if p is not handle]


async def verify_all_exited(
    handles: list[ProcessHandle], timeout: float = 10.0
) -> list[ProcessHandle]:
    """Return handles that are still alive after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    alive = [h for h in handles if h.is_running()]
    while alive and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
        alive = [h for h in handles if h.is_running()]
    return alive
