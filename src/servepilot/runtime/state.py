"""Runtime state file for ``servepilot status`` / ``servepilot stop``.

PIDs are validated against the recorded process creation time so a stale file never causes an
unrelated process (after PID reuse) to be signalled.
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
from pathlib import Path

import psutil
from pydantic import ValidationError

from servepilot.constants import RUNTIME_STATE_FILENAME, RUNTIME_STATE_SCHEMA_VERSION
from servepilot.exceptions import RuntimeStateError
from servepilot.fsutil import atomic_write_json, read_json
from servepilot.schemas.runtime import ChildProcessRecord, RuntimeState

CREATE_TIME_TOLERANCE_SECONDS = 2.0


def process_matches(pid: int, create_time: float) -> bool:
    """True if ``pid`` exists and was created at ``create_time`` (± tolerance)."""
    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        return abs(proc.create_time() - create_time) <= CREATE_TIME_TOLERANCE_SECONDS
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def current_process_create_time() -> float:
    return psutil.Process(os.getpid()).create_time()


class RuntimeStateStore:
    def __init__(self, state_dir: Path) -> None:
        self._dir = state_dir
        self.path = state_dir / RUNTIME_STATE_FILENAME

    def write(self, state: RuntimeState) -> None:
        atomic_write_json(self.path, state.model_dump(mode="json"))

    def read(self) -> RuntimeState | None:
        if not self.path.exists():
            return None
        try:
            raw = read_json(self.path)
        except (OSError, ValueError) as exc:
            raise RuntimeStateError(f"runtime state file {self.path} is unreadable: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != RUNTIME_STATE_SCHEMA_VERSION:
            raise RuntimeStateError(
                f"runtime state file {self.path} has an unsupported schema version",
                hints=[f"Delete {self.path} if no ServePilot deployment is running."],
            )
        try:
            return RuntimeState.model_validate(raw)
        except ValidationError as exc:
            raise RuntimeStateError(f"runtime state file {self.path} is invalid: {exc}") from exc

    def clear(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()

    def is_live(self, state: RuntimeState) -> bool:
        return process_matches(state.servepilot_pid, state.servepilot_create_time)

    def live_children(self, state: RuntimeState) -> list[ChildProcessRecord]:
        return [c for c in state.children if process_matches(c.pid, c.create_time)]

    def stop(self, state: RuntimeState, *, grace_seconds: float = 30.0) -> tuple[bool, list[str]]:
        """Stop a deployment: SIGTERM the ServePilot process (validated), then any live children.

        Returns ``(stopped, messages)``.
        """
        messages: list[str] = []
        if self.is_live(state):
            os.kill(state.servepilot_pid, signal.SIGTERM)
            messages.append(f"sent SIGTERM to ServePilot pid {state.servepilot_pid}")
            deadline = time.monotonic() + grace_seconds
            while time.monotonic() < deadline and self.is_live(state):
                time.sleep(0.25)
            if self.is_live(state):
                os.kill(state.servepilot_pid, signal.SIGKILL)
                messages.append(f"ServePilot pid {state.servepilot_pid} did not exit; sent SIGKILL")
                time.sleep(0.5)
        else:
            messages.append(
                f"ServePilot pid {state.servepilot_pid} is not running (stale state file)"
            )
        leftovers = self.live_children(state)
        for child in leftovers:
            try:
                pgid = os.getpgid(child.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                continue
            messages.append(f"sent SIGTERM to leftover engine pid {child.pid} ({child.replica_id})")
        if leftovers:
            deadline = time.monotonic() + grace_seconds
            while time.monotonic() < deadline and self.live_children(state):
                time.sleep(0.25)
            for child in self.live_children(state):
                try:
                    os.killpg(os.getpgid(child.pid), signal.SIGKILL)
                    messages.append(f"force-killed engine pid {child.pid}")
                except (ProcessLookupError, PermissionError):
                    continue
        stopped = not self.is_live(state) and not self.live_children(state)
        if stopped:
            self.clear()
        return stopped, messages
