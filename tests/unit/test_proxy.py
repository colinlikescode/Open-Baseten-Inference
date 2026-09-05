"""Proxy lifecycle regressions using an in-memory backend and ASGI requests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import Request
from starlette.types import Message

from servepilot.api.app import ServingContext
from servepilot.api.proxy import proxy_generation
from servepilot.runtime.router import ReplicaRouter
from servepilot.schemas.runtime import ReplicaStatus


@pytest.fixture
async def ctx() -> AsyncIterator[ServingContext]:
    router = ReplicaRouter(max_concurrency=1, max_queue_depth=0)
    router.add_replica("a", "http://backend.test")
    router.set_status("a", ReplicaStatus.HEALTHY)
    ctx = ServingContext(router=router, model_id="test", served_model_name="test")
    try:
        yield ctx
    finally:
        await ctx.aclose()


def request() -> Request:
    req = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "query_string": b"",
            "headers": [],
            "server": ("proxy.test", 80),
            "asgi": {"spec_version": "2.4"},
        }
    )
    req._body = b"{}"
    return req


async def assert_slot_available(ctx: ServingContext) -> None:
    assert ctx.router.inflight == 0
    assert ctx.metrics.requests_inflight._value.get() == 0
    async with asyncio.timeout(1):
        lease = await ctx.router.acquire()
    ctx.router.release(lease)


async def test_cancelled_before_backend_headers_releases_slot(ctx: ServingContext) -> None:
    sending = asyncio.Event()

    async def backend(_: httpx.Request) -> httpx.Response:
        sending.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    await ctx.http.aclose()
    ctx.http = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    task = asyncio.create_task(proxy_generation(request(), ctx, "chat_completions"))
    await asyncio.wait_for(sending.wait(), 1)
    assert ctx.router.inflight == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await assert_slot_available(ctx)
    assert ctx.router.total_errors == 0


class TrackedStream(httpx.AsyncByteStream):
    def __init__(self, *, close_error: bool = False) -> None:
        self.closed = False
        self.close_error = close_error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"data: hello\n\n"

    async def aclose(self) -> None:
        self.closed = True
        if self.close_error:
            raise httpx.ReadError("close failed")


@pytest.mark.parametrize("fail_on", ["http.response.start", "http.response.body", None])
async def test_response_cleanup_even_when_asgi_send_fails(
    ctx: ServingContext, fail_on: str | None
) -> None:
    stream = TrackedStream()
    await ctx.http.aclose()
    ctx.http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
    )
    req = request()
    response = await proxy_generation(req, ctx, "chat_completions")

    async def send(message: Message) -> None:
        if message["type"] == fail_on:
            raise RuntimeError("client disconnected")

    if fail_on:
        with pytest.raises(RuntimeError, match="client disconnected"):
            await response(req.scope, req.receive, send)
    else:
        await response(req.scope, req.receive, send)
    assert stream.closed
    await assert_slot_available(ctx)


async def test_backend_disconnect_propagates_without_retry(ctx: ServingContext) -> None:
    class BrokenStream(TrackedStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"data: partial\n\n"
            raise httpx.ReadError("backend disconnected")

    stream = BrokenStream()
    calls = 0

    def backend(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=stream)

    ctx.router.add_replica("b", "http://other.test")
    ctx.router.set_status("b", ReplicaStatus.HEALTHY)
    await ctx.http.aclose()
    ctx.http = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    req = request()
    response = await proxy_generation(req, ctx, "chat_completions")
    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    with pytest.raises(httpx.ReadError, match="backend disconnected"):
        await response(req.scope, req.receive, send)
    assert any(message.get("body") == b"data: partial\n\n" for message in messages)
    assert calls == 1 and stream.closed and ctx.router.total_errors == 1
    await assert_slot_available(ctx)


async def test_backend_close_error_still_releases_slot(ctx: ServingContext) -> None:
    stream = TrackedStream(close_error=True)
    await ctx.http.aclose()
    ctx.http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503, stream=stream))
    )
    with pytest.raises(httpx.ReadError, match="close failed"):
        await proxy_generation(request(), ctx, "chat_completions")
    assert stream.closed
    await assert_slot_available(ctx)
