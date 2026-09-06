"""Streaming pass-through proxy for OpenAI endpoints.

Requests are forwarded byte-for-byte (no schema re-interpretation) to the replica chosen by the
router. Responses are streamed incrementally; the in-flight slot is released when the stream ends,
the client disconnects or the backend drops. A request is retried on another replica at most once
and only if the first replica failed *before* returning any response body.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from servepilot.logging import get_logger
from servepilot.runtime.router import NoHealthyReplicaError, OverloadedError, RouteLease

if TYPE_CHECKING:
    from servepilot.api.app import ServingContext

log = get_logger(__name__)

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


class _ProxyResponse(StreamingResponse):
    """Own cleanup across the entire ASGI response, including failures sending headers."""

    def __init__(
        self,
        content: AsyncIterator[bytes],
        *,
        status_code: int,
        headers: dict[str, str],
        media_type: str | None,
        cleanup: Callable[[], Awaitable[None]],
    ) -> None:
        super().__init__(content, status_code=status_code, headers=headers, media_type=media_type)
        self._cleanup = cleanup

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._cleanup()


def error_response(status: int, message: str, error_type: str = "server_error") -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": error_type, "code": status}}, status_code=status
    )


def _forward_headers(request: Request) -> dict[str, str]:
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    headers["x-servepilot-proxy"] = "1"
    return headers


def _response_headers(upstream: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP}


def rewrite_model_field(body: bytes, served_name: str | None, backend_name: str | None) -> bytes:
    """Map the public model name to the backend's if they differ (keeps bytes untouched otherwise)."""
    if not served_name or not backend_name or served_name == backend_name:
        return body
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    if isinstance(payload, dict) and payload.get("model") == served_name:
        payload["model"] = backend_name
        return json.dumps(payload).encode("utf-8")
    return body


async def _send_with_retry(
    ctx: ServingContext, request: Request, lease: RouteLease, body: bytes, headers: dict[str, str]
) -> tuple[httpx.Response | None, str | None]:
    """Send to the leased replica; on a pre-body failure re-route once to another healthy replica.

    Returns ``(response, None)`` on success or ``(None, error)`` when every attempt failed; the
    caller releases the lease with that error so each failed attempt is charged exactly once.
    """
    router = ctx.router
    attempts = 0
    while True:
        replica = lease.replica
        url = f"{replica.base_url}{request.url.path}"
        if request.url.query:
            url += f"?{request.url.query}"
        upstream_request = ctx.http.build_request(
            request.method, url, content=body, headers=headers
        )
        try:
            upstream = await ctx.http.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.warning("replica %s failed before responding: %s", replica.id, error)
        else:
            if upstream.status_code < 500:
                return upstream, None
            error = f"HTTP {upstream.status_code}"
            await upstream.aclose()
        if attempts >= 1:
            return None, error
        attempts += 1
        try:
            router.reroute(lease, error=error)
        except NoHealthyReplicaError:
            return None, error


async def proxy_generation(request: Request, ctx: ServingContext, endpoint: str) -> Response:
    router = ctx.router
    metrics = ctx.metrics
    body = await request.body()
    body = rewrite_model_field(body, ctx.served_model_name, ctx.backend_model_name)
    started = time.perf_counter()
    metrics.requests_inflight.inc()
    handed_off = False
    lease: RouteLease | None = None
    upstream: httpx.Response | None = None
    error: str | None = None

    async def cleanup() -> None:
        try:
            if upstream is not None:
                await upstream.aclose()
        finally:
            # Accounting must balance even if closing the connection raises or is cancelled.
            if lease is not None:
                router.release(lease, failed=error is not None, error=error)
            metrics.requests_inflight.dec()
            metrics.request_latency.labels(endpoint=endpoint).observe(time.perf_counter() - started)

    try:
        try:
            lease = await router.acquire({"endpoint": endpoint})
        except OverloadedError as exc:
            metrics.request_errors_total.labels(endpoint=endpoint, reason="overloaded").inc()
            metrics.requests_total.labels(endpoint=endpoint, status_class="5xx").inc()
            return error_response(503, f"ServePilot is overloaded: {exc}", "overloaded_error")
        except NoHealthyReplicaError:
            metrics.request_errors_total.labels(
                endpoint=endpoint, reason="no_healthy_replica"
            ).inc()
            metrics.requests_total.labels(endpoint=endpoint, status_class="5xx").inc()
            return error_response(
                503, "no healthy model replica is available", "service_unavailable"
            )

        upstream, error = await _send_with_retry(
            ctx, request, lease, body, _forward_headers(request)
        )
        if upstream is None:
            metrics.request_errors_total.labels(
                endpoint=endpoint, reason="backend_unreachable"
            ).inc()
            metrics.requests_total.labels(endpoint=endpoint, status_class="5xx").inc()
            return error_response(502, "model replica is unreachable or failing", "bad_gateway")

        metrics.requests_total.labels(
            endpoint=endpoint, status_class=f"{upstream.status_code // 100}xx"
        ).inc()
        response_headers = _response_headers(upstream)
        response_headers["x-servepilot-replica"] = lease.replica.id
        backend_response = upstream
        active_lease = lease

        async def body_stream() -> AsyncIterator[bytes]:
            nonlocal error
            try:
                async for chunk in backend_response.aiter_raw():
                    yield chunk
            except httpx.HTTPError as exc:
                # Backend dropped mid-stream: output may already have been delivered, never retry.
                error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "replica %s dropped the connection mid-response: %s",
                    active_lease.replica.id,
                    error,
                )
                metrics.request_errors_total.labels(
                    endpoint=endpoint, reason="backend_disconnect"
                ).inc()
                raise

        response = _ProxyResponse(
            body_stream(),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
            cleanup=cleanup,
        )
        handed_off = True
        return response
    finally:
        if not handed_off:
            await cleanup()
