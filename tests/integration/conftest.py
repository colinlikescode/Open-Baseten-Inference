"""Fixtures that spawn fake OpenAI backends and an in-process ServePilot router."""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field

import httpx
import pytest

from servepilot.api.app import ServingContext, create_app
from servepilot.runtime.ports import ephemeral_port
from servepilot.runtime.router import ReplicaRouter
from servepilot.runtime.server import HTTPServer
from servepilot.schemas.runtime import ReplicaStatus


@dataclass
class FakeBackend:
    port: int
    proc: subprocess.Popen[bytes]
    ttft_ms: float
    tpot_ms: float

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stats(self) -> dict[str, object]:
        return httpx.get(f"{self.base_url}/_fake/stats", timeout=5).json()  # type: ignore[no-any-return]

    def configure(self, **values: object) -> None:
        httpx.post(f"{self.base_url}/_fake/config", json=values, timeout=5).raise_for_status()

    def crash(self) -> None:
        with contextlib.suppress(httpx.HTTPError):
            httpx.post(f"{self.base_url}/_fake/crash", timeout=5)
        self.proc.wait(timeout=10)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def start_backend(
    *,
    ttft_ms: float = 5.0,
    tpot_ms: float = 1.0,
    capacity: int = 64,
    model: str = "fake-model",
    error_rate: float = 0.0,
    extra: list[str] | None = None,
) -> FakeBackend:
    port = ephemeral_port()
    cmd = [
        sys.executable,
        "-m",
        "servepilot.testing.fake_openai",
        "--port",
        str(port),
        "--model",
        model,
        "--ttft-ms",
        str(ttft_ms),
        "--tpot-ms",
        str(tpot_ms),
        "--capacity",
        str(capacity),
        "--error-rate",
        str(error_rate),
        *(extra or []),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    backend = FakeBackend(port=port, proc=proc, ttft_ms=ttft_ms, tpot_ms=tpot_ms)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if httpx.get(f"{backend.base_url}/health", timeout=1).status_code == 200:
                return backend
        except httpx.HTTPError:
            time.sleep(0.1)
        if proc.poll() is not None:
            raise RuntimeError("fake backend exited during startup")
    proc.kill()
    raise RuntimeError("fake backend did not start")


@pytest.fixture
def two_backends() -> Iterator[tuple[FakeBackend, FakeBackend]]:
    a = start_backend()
    b = start_backend()
    try:
        yield a, b
    finally:
        a.stop()
        b.stop()


@dataclass
class RouterHarness:
    server: HTTPServer
    router: ReplicaRouter
    ctx: ServingContext
    backends: list[FakeBackend] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return self.server.base_url


@pytest.fixture
async def harness(two_backends: tuple[FakeBackend, FakeBackend]) -> AsyncIterator[RouterHarness]:
    router = ReplicaRouter(max_concurrency=None)
    for i, backend in enumerate(two_backends):
        router.add_replica(f"replica-{i}", backend.base_url, gpu_ids=[i])
        router.set_status(f"replica-{i}", ReplicaStatus.HEALTHY)
    ctx = ServingContext(
        router=router,
        model_id="fake-model",
        served_model_name="fake-model",
        backend_model_name="fake-model",
    )
    server = HTTPServer(create_app(ctx), "127.0.0.1", ephemeral_port())
    await server.start()
    try:
        yield RouterHarness(server=server, router=router, ctx=ctx, backends=list(two_backends))
    finally:
        await server.stop()
        await ctx.aclose()


async def wait_until(predicate, timeout: float = 10.0, interval: float = 0.05) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition not met in time")
