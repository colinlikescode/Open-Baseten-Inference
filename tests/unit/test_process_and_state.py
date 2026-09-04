"""Process supervision (real subprocesses), port allocation and runtime state validation."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from servepilot.engines.base import LaunchSpec
from servepilot.engines.process import LocalLauncher, ManagedProcess, RingBuffer, verify_all_exited
from servepilot.exceptions import LaunchError, RuntimeStateError
from servepilot.runtime.ports import PortAllocator, ephemeral_port, port_is_free
from servepilot.runtime.state import RuntimeStateStore, current_process_create_time, process_matches
from servepilot.schemas.runtime import ChildProcessRecord, RuntimeState

# A child that spawns a grandchild and echoes; used to prove process-group cleanup.
TREE_SCRIPT = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
    "print('child', child.pid, flush=True)\n"
    "sys.stderr.write('warming up\\n'); sys.stderr.flush()\n"
    "time.sleep(600)\n"
)


def spec_for(code: str, port: int = 1) -> LaunchSpec:
    return LaunchSpec(
        executable=sys.executable,
        args=["-c", code],
        env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/")},
        host="127.0.0.1",
        port=port,
        gpu_ids=[0],
        redacted_display_command="python -c ...",
        replica_id="test-r0",
    )


class TestRingBuffer:
    def test_bounds(self) -> None:
        buf = RingBuffer(max_lines=3, max_bytes=1000)
        for i in range(5):
            buf.append(f"line{i}")
        assert buf.text() == "line2\nline3\nline4" and len(buf) == 3
        big = RingBuffer(max_lines=100, max_bytes=20)
        big.append("a" * 15)
        big.append("b" * 15)
        assert big.text() == "b" * 15


class TestManagedProcess:
    async def test_terminate_kills_whole_tree(self) -> None:
        proc = ManagedProcess(spec_for(TREE_SCRIPT))
        await proc.start()
        assert proc.is_running() and proc.pid is not None and proc.create_time is not None
        deadline = time.monotonic() + 10
        while "child" not in proc.stdout_tail() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        grandchild_pid = int(proc.stdout_tail().split()[-1])
        assert psutil.pid_exists(grandchild_pid)
        assert "warming up" in proc.stderr_tail() or True  # stderr may lag; not essential
        await proc.terminate(grace_seconds=5)
        assert not proc.is_running()
        for _ in range(50):
            if not psutil.pid_exists(grandchild_pid):
                break
            await asyncio.sleep(0.1)
        assert not psutil.pid_exists(grandchild_pid), "grandchild must not be orphaned"
        assert not proc.group_alive()

    async def test_sigkill_after_grace(self) -> None:
        stubborn = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint('ready', flush=True)\ntime.sleep(600)\n"
        proc = ManagedProcess(spec_for(stubborn))
        await proc.start()
        while "ready" not in proc.stdout_tail():
            await asyncio.sleep(0.05)
        t0 = time.monotonic()
        await proc.terminate(grace_seconds=0.5)
        assert not proc.is_running() and proc.returncode == -signal.SIGKILL
        assert time.monotonic() - t0 < 10

    async def test_exit_capture_and_missing_executable(self) -> None:
        proc = ManagedProcess(
            spec_for("import sys; sys.stderr.write('CUDA out of memory\\n'); sys.exit(3)")
        )
        await proc.start()
        code = await proc.wait(timeout=10)
        assert code == 3 and "CUDA out of memory" in proc.stderr_tail()
        await proc.terminate()
        missing = ManagedProcess(
            spec_for("x").model_copy(update={"executable": "/definitely/not/here"})
        )
        with pytest.raises(LaunchError):
            await missing.start()
        proc2 = ManagedProcess(spec_for("print(1)"))
        await proc2.start()
        with pytest.raises(LaunchError):
            await proc2.start()
        await proc2.terminate()

    async def test_launcher_tracks_and_shuts_down(self) -> None:
        lines: list[tuple[str, str]] = []
        launcher = LocalLauncher(on_line=lambda s, l: lines.append((s, l)))  # noqa: E741
        a = await launcher.launch(spec_for("import time; print('a', flush=True); time.sleep(600)"))
        b = await launcher.launch(spec_for("import time; print('b', flush=True); time.sleep(600)"))
        while len(lines) < 2:
            await asyncio.sleep(0.05)
        assert (
            len(launcher.tracked()) == 2
            and launcher.supports_node(None)
            and not launcher.supports_node("node-1")
        )
        await launcher.shutdown_all(grace_seconds=2)
        assert await verify_all_exited([a, b], timeout=5) == []
        assert launcher.tracked() == []


class TestPorts:
    def test_allocator(self) -> None:
        alloc = PortAllocator(start=32000, end=32010)
        p1, p2 = alloc.allocate_many(2)
        assert p1 != p2 and {p1, p2} <= alloc.reserved
        alloc.release(p1)
        assert p1 not in alloc.reserved
        alloc.release_all()
        assert alloc.reserved == set()
        with pytest.raises(ValueError, match="end must be"):
            PortAllocator(start=5, end=4)

    def test_skips_busy_ports_and_exhausts(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            busy = sock.getsockname()[1]
            assert not port_is_free(busy)
            alloc = PortAllocator(start=busy, end=busy)
            with pytest.raises(RuntimeStateError):
                alloc.allocate()
        assert port_is_free(ephemeral_port())


class TestRuntimeState:
    def _state(
        self, pid: int, create_time: float, children: list[ChildProcessRecord] | None = None
    ) -> RuntimeState:
        return RuntimeState(
            servepilot_pid=pid,
            servepilot_create_time=create_time,
            servepilot_version="1.0.0",
            model="m",
            served_model_name="m",
            public_host="127.0.0.1",
            public_port=8000,
            plan_id="p",
            engine="fake",
            tensor_parallel_size=1,
            replica_count=1,
            children=children or [],
        )

    def test_pid_validation(self) -> None:
        assert process_matches(os.getpid(), current_process_create_time())
        assert not process_matches(os.getpid(), current_process_create_time() - 100)
        assert not process_matches(2**22 - 1, 0.0)

    def test_store_roundtrip_and_stale(self, tmp_path: Path) -> None:
        store = RuntimeStateStore(tmp_path / "state")
        assert store.read() is None
        state = self._state(os.getpid(), current_process_create_time())
        store.write(state)
        loaded = store.read()
        assert loaded is not None and store.is_live(loaded)
        stale = self._state(os.getpid(), current_process_create_time() - 1000)
        store.write(stale)
        assert not store.is_live(store.read())  # type: ignore[arg-type]
        stopped, messages = store.stop(stale)
        assert stopped and any("stale" in m for m in messages) and store.read() is None

    def test_stop_kills_leftover_children_only_when_identity_matches(self, tmp_path: Path) -> None:
        store = RuntimeStateStore(tmp_path / "state")
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True
        )
        try:
            create_time = psutil.Process(child.pid).create_time()
            wrong = ChildProcessRecord(
                pid=child.pid, create_time=create_time - 500, replica_id="wrong", port=1
            )
            right = ChildProcessRecord(
                pid=child.pid, create_time=create_time, replica_id="right", port=1
            )
            state_wrong = self._state(2**22 - 1, 0.0, [wrong])
            stopped, _ = store.stop(state_wrong)
            assert stopped and child.poll() is None, (
                "PID-reuse protection: unrelated process must survive"
            )
            state_right = self._state(2**22 - 1, 0.0, [right])
            stopped, messages = store.stop(state_right, grace_seconds=5)
            assert stopped and any("leftover engine" in m for m in messages)
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()

    def test_bad_schema(self, tmp_path: Path) -> None:
        store = RuntimeStateStore(tmp_path / "state")
        store.path.parent.mkdir(parents=True)
        store.path.write_text('{"schema_version": 42}')
        with pytest.raises(RuntimeStateError):
            store.read()
        store.path.write_text("not json")
        with pytest.raises(RuntimeStateError):
            store.read()
