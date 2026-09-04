"""FastAPI application exposing the OpenAI-compatible endpoint plus ServePilot introspection."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from servepilot.api.health import health_payload
from servepilot.api.metrics import RouterMetrics
from servepilot.api.proxy import proxy_generation
from servepilot.api.status import status_payload
from servepilot.constants import DEFAULT_PROXY_TIMEOUT_SECONDS
from servepilot.hardware.base import HardwareProvider
from servepilot.runtime.router import ReplicaRouter
from servepilot.schemas.plan import SelectedPlan


@dataclass
class ServingContext:
    """Everything the HTTP layer needs; owned by the deployment (or a tuning session)."""

    router: ReplicaRouter
    model_id: str
    served_model_name: str
    backend_model_name: str | None = None
    selected: SelectedPlan | None = None
    hardware: HardwareProvider | None = None
    gpu_indices: list[int] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    metrics: RouterMetrics = field(default_factory=RouterMetrics)
    http: httpx.AsyncClient = field(
        default_factory=lambda: httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_PROXY_TIMEOUT_SECONDS, connect=10.0),
            limits=httpx.Limits(max_connections=4096, max_keepalive_connections=1024),
        )
    )
    extra_status: Callable[[], dict[str, Any]] | None = None
    inflight_streams_started: int = 0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at

    async def aclose(self) -> None:
        await self.http.aclose()


def create_app(ctx: ServingContext) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await ctx.aclose()

    app = FastAPI(
        title="ServePilot", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    app.state.ctx = ctx

    @app.get("/health")
    async def health() -> Response:
        payload, healthy = health_payload(ctx)
        return JSONResponse(payload, status_code=200 if healthy else 503)

    @app.get("/status")
    async def status() -> Response:
        return JSONResponse(status_payload(ctx))

    @app.get("/metrics")
    async def metrics() -> Response:
        ctx.metrics.refresh_router(ctx.router)
        await ctx.metrics.refresh_gpus(ctx.hardware, ctx.gpu_indices)
        body, content_type = ctx.metrics.render()
        return Response(content=body, media_type=content_type)

    @app.get("/v1/models")
    async def models() -> Response:
        entry: dict[str, Any] = {
            "id": ctx.served_model_name,
            "object": "model",
            "created": int(ctx.started_at),
            "owned_by": "servepilot",
            "root": ctx.model_id,
        }
        if ctx.selected is not None:
            entry["max_model_len"] = ctx.selected.plan.context_length
        return JSONResponse({"object": "list", "data": [entry]})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await proxy_generation(request, ctx, "chat_completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await proxy_generation(request, ctx, "completions")

    return app
