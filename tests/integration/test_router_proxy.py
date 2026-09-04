"""Router + streaming proxy against two real fake OpenAI backends."""

from __future__ import annotations

import asyncio
import itertools
import json
import time

import httpx
import pytest

from servepilot.runtime.health import HealthChecker
from servepilot.runtime.replicas import Replica, ReplicaSet
from servepilot.schemas.runtime import ReplicaStatus
from tests.integration.conftest import FakeBackend, RouterHarness, start_backend, wait_until

CHAT = {
    "model": "fake-model",
    "messages": [{"role": "user", "content": "Hello there"}],
    "max_tokens": 8,
}


async def test_requests_are_distributed_across_replicas(harness: RouterHarness) -> None:
    async with httpx.AsyncClient(base_url=harness.base_url, timeout=30) as client:
        responses = await asyncio.gather(
            *(client.post("/v1/chat/completions", json=CHAT) for _ in range(20))
        )
    assert all(r.status_code == 200 for r in responses)
    replicas = {r.headers["x-servepilot-replica"] for r in responses}
    assert replicas == {"replica-0", "replica-1"}
    stats = [b.stats()["total_requests"] for b in harness.backends]
    assert min(stats) >= 5, f"traffic must reach both backends: {stats}"
    body = responses[0].json()
    assert (
        body["choices"][0]["message"]["content"].startswith("tok0")
        and body["usage"]["completion_tokens"] == 8
    )
    assert harness.router.inflight == 0


async def test_least_inflight_prefers_idle_backend(harness: RouterHarness) -> None:
    slow, _fast = harness.backends
    slow.configure(tpot_ms=200.0)  # backend A now takes ~4 s per 20-token generation
    a = harness.router.replica("replica-0")
    b = harness.router.replica("replica-1")
    assert a is not None and b is not None
    async with httpx.AsyncClient(base_url=harness.base_url, timeout=60) as client:
        long_requests = [
            asyncio.create_task(
                client.post("/v1/chat/completions", json={**CHAT, "max_tokens": 20})
            )
            for _ in range(3)
        ]
        # A keeps at least one long generation in flight while B drains its share quickly.
        await wait_until(lambda: a.inflight_requests >= 1 and b.inflight_requests == 0, timeout=15)
        quick = [
            await client.post("/v1/chat/completions", json={**CHAT, "max_tokens": 1})
            for _ in range(6)
        ]
        picks = [r.headers["x-servepilot-replica"] for r in quick]
        assert picks == ["replica-1"] * 6, picks
        await asyncio.gather(*long_requests)


async def test_streaming_is_incremental(harness: RouterHarness) -> None:
    harness.backends[0].configure(tpot_ms=40.0)
    harness.backends[1].configure(tpot_ms=40.0)
    payload = {**CHAT, "max_tokens": 10, "stream": True, "stream_options": {"include_usage": True}}
    arrivals: list[float] = []
    chunks: list[dict[str, object]] = []
    async with (
        httpx.AsyncClient(base_url=harness.base_url, timeout=60) as client,
        client.stream("POST", "/v1/chat/completions", json=payload) as resp,
    ):
        assert resp.status_code == 200 and resp.headers["content-type"].startswith(
            "text/event-stream"
        )
        async for line in resp.aiter_lines():
            if line.startswith("data:") and "[DONE]" not in line:
                arrivals.append(time.perf_counter())
                chunks.append(json.loads(line[5:]))
    assert len(chunks) >= 11  # 10 content chunks + usage chunk
    gaps = [b - a for a, b in itertools.pairwise(arrivals)]
    assert max(gaps) >= 0.02, "chunks must arrive over time, not all at once"
    assert (arrivals[-1] - arrivals[0]) >= 0.25, "full stream must not be buffered before delivery"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c.get("choices"))  # type: ignore[index]
    assert text.startswith("tok0 tok1")
    assert chunks[-1]["usage"]["completion_tokens"] == 10  # type: ignore[index]
    await wait_until(lambda: harness.router.inflight == 0)


async def test_completions_endpoint_and_models(harness: RouterHarness) -> None:
    async with httpx.AsyncClient(base_url=harness.base_url, timeout=30) as client:
        resp = await client.post(
            "/v1/completions", json={"model": "fake-model", "prompt": "Once upon", "max_tokens": 3}
        )
        assert resp.status_code == 200 and resp.json()["choices"][0]["text"].startswith("tok0")
        models = (await client.get("/v1/models")).json()
        assert (
            models["data"][0]["id"] == "fake-model"
            and models["data"][0]["owned_by"] == "servepilot"
        )


async def test_health_status_metrics(harness: RouterHarness) -> None:
    async with httpx.AsyncClient(base_url=harness.base_url, timeout=30) as client:
        await client.post("/v1/chat/completions", json=CHAT)
        health = await client.get("/health")
        assert health.status_code == 200 and health.json() == {
            "status": "healthy",
            "model": "fake-model",
            "healthy_replicas": 2,
            "total_replicas": 2,
        }
        status = (await client.get("/status")).json()
        assert status["router"]["total_requests"] == 1 and status["router"]["healthy_replicas"] == 2
        metrics = (await client.get("/metrics")).text
        for name in (
            "servepilot_requests_total",
            "servepilot_requests_inflight",
            "servepilot_router_queue_depth",
            "servepilot_replica_health",
            "servepilot_replica_inflight",
            "servepilot_request_latency_seconds",
        ):
            assert name in metrics
        assert 'servepilot_replica_health{replica="replica-0"} 1.0' in metrics


async def test_failed_backend_is_excluded_and_service_stays_up(harness: RouterHarness) -> None:
    dead, _alive = harness.backends
    dead.crash()
    # Simulate what the health checker does: mark it unhealthy after the process died.
    harness.router.set_status("replica-0", ReplicaStatus.UNHEALTHY, "process exited")
    async with httpx.AsyncClient(base_url=harness.base_url, timeout=30) as client:
        responses = await asyncio.gather(
            *(client.post("/v1/chat/completions", json=CHAT) for _ in range(6))
        )
        assert all(
            r.status_code == 200 and r.headers["x-servepilot-replica"] == "replica-1"
            for r in responses
        )
        health = await client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "degraded"
        harness.router.set_status("replica-1", ReplicaStatus.UNHEALTHY)
        down = await client.post("/v1/chat/completions", json=CHAT)
        assert down.status_code == 503 and down.json()["error"]["type"] == "service_unavailable"
        assert (await client.get("/health")).status_code == 503


async def test_pre_body_failure_is_retried_once_on_another_replica(harness: RouterHarness) -> None:
    dead, _alive = harness.backends
    dead.crash()  # replica-0 is still marked healthy: the connection failure must trigger one retry
    async with httpx.AsyncClient(base_url=harness.base_url, timeout=30) as client:
        responses = await asyncio.gather(
            *(client.post("/v1/chat/completions", json=CHAT) for _ in range(4))
        )
    assert all(
        r.status_code == 200 and r.headers["x-servepilot-replica"] == "replica-1" for r in responses
    )
    assert harness.router.replica("replica-0").failures >= 1  # type: ignore[union-attr]


async def test_backend_5xx_before_body_is_retried(harness: RouterHarness) -> None:
    flaky, good = harness.backends
    flaky.configure(error_rate=1.0)
    async with httpx.AsyncClient(base_url=harness.base_url, timeout=30) as client:
        responses = await asyncio.gather(
            *(client.post("/v1/chat/completions", json=CHAT) for _ in range(6))
        )
    assert all(
        r.status_code == 200 and r.headers["x-servepilot-replica"] == "replica-1" for r in responses
    )
    good.configure(error_rate=1.0)
    async with httpx.AsyncClient(base_url=harness.base_url, timeout=30) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT)
    assert resp.status_code == 502 and resp.json()["error"]["type"] == "bad_gateway"


async def test_backpressure_queue_and_overload(harness: RouterHarness) -> None:
    harness.router.set_max_concurrency(2)
    harness.router._max_queue_depth = 2
    for b in harness.backends:
        b.configure(tpot_ms=100.0)
    async with httpx.AsyncClient(base_url=harness.base_url, timeout=60) as client:
        tasks = [
            asyncio.create_task(
                client.post("/v1/chat/completions", json={**CHAT, "max_tokens": 10})
            )
            for _ in range(6)
        ]
        await asyncio.sleep(0.3)
        status = (await client.get("/status")).json()
        assert status["router"]["inflight"] <= 2
        responses = await asyncio.gather(*tasks)
    codes = sorted(r.status_code for r in responses)
    assert codes.count(200) == 4 and codes.count(503) == 2, codes
    overloaded = next(r for r in responses if r.status_code == 503)
    assert overloaded.json()["error"]["type"] == "overloaded_error"
    async with httpx.AsyncClient(base_url=harness.base_url, timeout=30) as client:
        metrics = (await client.get("/metrics")).text
    assert (
        'servepilot_request_errors_total{endpoint="chat_completions",reason="overloaded"} 2.0'
        in metrics
    )


async def test_client_disconnect_releases_inflight(harness: RouterHarness) -> None:
    harness.backends[0].configure(tpot_ms=200.0)
    harness.backends[1].configure(tpot_ms=200.0)
    payload = {**CHAT, "max_tokens": 50, "stream": True}
    async with httpx.AsyncClient(base_url=harness.base_url, timeout=60) as client:
        async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            async for _ in resp.aiter_lines():
                break  # disconnect after the first chunk
        await wait_until(lambda: harness.router.inflight == 0, timeout=15)


async def test_openai_sdk_compatibility(harness: RouterHarness) -> None:
    openai = pytest.importorskip("openai")
    client = openai.AsyncOpenAI(base_url=f"{harness.base_url}/v1", api_key="not-needed")
    completion = await client.chat.completions.create(
        model="fake-model", messages=[{"role": "user", "content": "Hello"}], max_tokens=4
    )
    assert completion.choices[0].message.content is not None and completion.choices[
        0
    ].message.content.startswith("tok0")
    assert completion.usage is not None and completion.usage.completion_tokens == 4
    stream = await client.chat.completions.create(
        model="fake-model", messages=[{"role": "user", "content": "Hi"}], max_tokens=3, stream=True
    )
    pieces = [
        chunk.choices[0].delta.content
        async for chunk in stream
        if chunk.choices and chunk.choices[0].delta.content
    ]
    assert pieces == ["tok0", " tok1", " tok2"]
    models = await client.models.list()
    assert models.data[0].id == "fake-model"
    await client.close()


async def test_served_model_name_rewrite(two_backends: tuple[FakeBackend, FakeBackend]) -> None:
    from servepilot.api.app import ServingContext, create_app
    from servepilot.runtime.ports import ephemeral_port
    from servepilot.runtime.router import ReplicaRouter
    from servepilot.runtime.server import HTTPServer

    router = ReplicaRouter()
    router.add_replica("r0", two_backends[0].base_url)
    router.set_status("r0", ReplicaStatus.HEALTHY)
    ctx = ServingContext(
        router=router,
        model_id="org/real-model",
        served_model_name="my-model",
        backend_model_name="fake-model",
    )
    server = HTTPServer(create_app(ctx), "127.0.0.1", ephemeral_port())
    await server.start()
    try:
        async with httpx.AsyncClient(base_url=server.base_url, timeout=30) as client:
            resp = await client.post("/v1/chat/completions", json={**CHAT, "model": "my-model"})
            assert resp.status_code == 200 and resp.json()["model"] == "fake-model"
            models = (await client.get("/v1/models")).json()
            assert (
                models["data"][0]["id"] == "my-model"
                and models["data"][0]["root"] == "org/real-model"
            )
    finally:
        await server.stop()
        await ctx.aclose()


class _StaticEngineStub:
    """Just enough of an engine for HealthChecker/ReplicaSet restart tests."""

    def version(self) -> str:
        return "stub"


async def test_health_checker_marks_dead_replica_and_keeps_others(harness: RouterHarness) -> None:
    from servepilot.engines.base import LaunchSpec

    class _Handle:
        def __init__(self, alive: bool) -> None:
            self._alive = alive
            self.returncode = None if alive else 1
            self.spec = None

        def is_running(self) -> bool:
            return self._alive

    replicas = []
    for i, backend in enumerate(harness.backends):
        spec = LaunchSpec(
            executable="x",
            args=[],
            env={},
            host="127.0.0.1",
            port=backend.port,
            gpu_ids=[i],
            redacted_display_command="x",
            replica_id=f"replica-{i}",
        )
        replicas.append(Replica(index=i, spec=spec, process=_Handle(alive=True), ready=True))  # type: ignore[arg-type]
    replica_set = ReplicaSet.__new__(ReplicaSet)
    replica_set.replicas = replicas
    checker = HealthChecker(
        harness.router,
        replica_set,
        interval=0.1,
        timeout=2.0,
        unhealthy_after=2,
        restart_enabled=False,
    )
    async with httpx.AsyncClient(timeout=5) as client:
        await checker.check_once(client)
        assert all(r.status == ReplicaStatus.HEALTHY for r in harness.router.replicas)
        harness.backends[0].crash()
        await checker.check_once(client)
        await checker.check_once(client)
        assert harness.router.replica("replica-0").status == ReplicaStatus.UNHEALTHY  # type: ignore[union-attr]
        assert harness.router.replica("replica-1").status == ReplicaStatus.HEALTHY  # type: ignore[union-attr]
        assert any("unhealthy" in e for e in checker.events)
        replicas[1].process = _Handle(alive=False)  # type: ignore[assignment]
        await checker.check_once(client)
        assert harness.router.replica("replica-1").status == ReplicaStatus.STOPPED  # type: ignore[union-attr]


async def test_start_backend_helper_reports_stats() -> None:
    backend = start_backend(ttft_ms=1, tpot_ms=1)
    try:
        assert backend.stats()["total_requests"] == 0
    finally:
        backend.stop()
