"""Streaming pass-through proxy for OpenAI endpoints.

Requests are forwarded byte-for-byte (no schema re-interpretation) to the replica chosen by the
router. Responses are streamed incrementally; the in-flight slot is released when the stream ends,
the client disconnects or the backend drops. A request is retried on another replica at most once
and only if the first replica failed *before* returning any response body.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

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
) -> httpx.Response | None:
    """Send to the leased replica; on a pre-body failure re-route once to another healthy replica."""
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
        upstream: httpx.Response | None
        try:
            upstream = await ctx.http.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            log.warning("replica %s failed before responding: %s", replica.id, exc)
            router.mark_failure(replica.id, str(exc))
            upstream = None
        if upstream is not None and upstream.status_code < 500:
            return upstream
        if upstream is not None:
            router.mark_failure(replica.id, f"HTTP {upstream.status_code}")
            await upstream.aclose()
        if attempts >= 1:
            return None
        attempts += 1
        try:
            router.reroute(lease)
        except NoHealthyReplicaError:
            return None


async def proxy_generation(request: Request, ctx: ServingContext, endpoint: str) -> Response:
    router = ctx.router
    metrics = ctx.metrics
    body = await request.body()
    body = rewrite_model_field(body, ctx.served_model_name, ctx.backend_model_name)
    started = time.perf_counter()
    metrics.requests_inflight.inc()
    handed_off = False
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

        upstream = await _send_with_retry(ctx, request, lease, body, _forward_headers(request))
        if upstream is None:
            router.release(lease, failed=True)
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

        async def body_stream() -> AsyncIterator[bytes]:
            failed = False
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            except httpx.HTTPError as exc:
                # Backend dropped mid-stream: output may already have been delivered, never retry.
                failed = True
                log.warning(
                    "replica %s dropped the connection mid-response: %s", lease.replica.id, exc
                )
                metrics.request_errors_total.labels(
                    endpoint=endpoint, reason="backend_disconnect"
                ).inc()
            finally:
                await upstream.aclose()
                router.release(lease, failed=failed)
                metrics.requests_inflight.dec()
                metrics.request_latency.labels(endpoint=endpoint).observe(
                    time.perf_counter() - started
                )

        handed_off = True
        return StreamingResponse(
            body_stream(),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )
    finally:
        if not handed_off:
            metrics.requests_inflight.dec()
            metrics.request_latency.labels(endpoint=endpoint).observe(time.perf_counter() - started)
