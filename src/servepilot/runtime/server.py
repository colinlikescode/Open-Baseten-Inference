"""In-process uvicorn server helper (signals are handled by ServePilot, not uvicorn)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator

import uvicorn
from fastapi import FastAPI

from servepilot.exceptions import RuntimeStateError


class _QuietServer(uvicorn.Server):
    """uvicorn server that never installs its own signal handlers."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


class HTTPServer:
    def __init__(self, app: FastAPI, host: str, port: int) -> None:
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
            loop="asyncio",
            lifespan="on",
            timeout_graceful_shutdown=10,
            h11_max_incomplete_event_size=16 * 1024 * 1024,
        )
        self._server = _QuietServer(config)
        self._task: asyncio.Task[None] | None = None
        self.host = host
        self.port = port

    async def start(self, timeout: float = 30.0) -> None:
        self._task = asyncio.create_task(self._server.serve())
        deadline = asyncio.get_running_loop().time() + timeout
        while not self._server.started:
            if self._task.done():
                exc = self._task.exception()
                raise RuntimeStateError(
                    f"could not start the ServePilot HTTP server on {self.host}:{self.port}: {exc}",
                    hints=["Is another process using the port? Choose a different --port."],
                )
            if asyncio.get_running_loop().time() > deadline:
                raise RuntimeStateError(
                    f"ServePilot HTTP server did not start within {timeout:.0f}s"
                )
            await asyncio.sleep(0.02)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"
