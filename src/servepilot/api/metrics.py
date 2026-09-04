"""Prometheus metrics for the router.

Labels are limited to low-cardinality values (endpoint, status class, replica id, GPU index);
request ids or prompts are never used as labels.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from servepilot.hardware.base import HardwareProvider
from servepilot.runtime.router import ReplicaRouter
from servepilot.schemas.runtime import ReplicaStatus


class RouterMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "servepilot_requests_total",
            "Requests proxied by ServePilot",
            ["endpoint", "status_class"],
            registry=self.registry,
        )
        self.requests_inflight = Gauge(
            "servepilot_requests_inflight",
            "Requests currently being proxied",
            registry=self.registry,
        )
        self.request_errors_total = Counter(
            "servepilot_request_errors_total",
            "Requests that failed",
            ["endpoint", "reason"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "servepilot_request_latency_seconds",
            "End-to-end request latency as seen by the router",
            ["endpoint"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600),
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "servepilot_router_queue_depth",
            "Requests waiting for a concurrency slot",
            registry=self.registry,
        )
        self.replica_health = Gauge(
            "servepilot_replica_health",
            "1 when the replica is healthy",
            ["replica"],
            registry=self.registry,
        )
        self.replica_inflight = Gauge(
            "servepilot_replica_inflight",
            "In-flight requests per replica",
            ["replica"],
            registry=self.registry,
        )
        self.gpu_memory_used = Gauge(
            "servepilot_gpu_memory_used_bytes", "GPU memory used", ["gpu"], registry=self.registry
        )
        self.gpu_utilization = Gauge(
            "servepilot_gpu_utilization_ratio",
            "GPU utilization (0-1)",
            ["gpu"],
            registry=self.registry,
        )

    def refresh_router(self, router: ReplicaRouter) -> None:
        self.queue_depth.set(router.queue_depth)
        for r in router.replicas:
            self.replica_health.labels(replica=r.id).set(
                1 if r.status == ReplicaStatus.HEALTHY else 0
            )
            self.replica_inflight.labels(replica=r.id).set(r.inflight_requests)

    async def refresh_gpus(
        self, hardware: HardwareProvider | None, gpu_indices: Sequence[int]
    ) -> None:
        if hardware is None or not gpu_indices:
            return
        try:
            samples = await asyncio.to_thread(hardware.sample, list(gpu_indices))
        except Exception:
            return
        for s in samples:
            if s.memory_used_bytes is not None:
                self.gpu_memory_used.labels(gpu=str(s.index)).set(s.memory_used_bytes)
            if s.utilization_percent is not None:
                self.gpu_utilization.labels(gpu=str(s.index)).set(s.utilization_percent / 100.0)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
