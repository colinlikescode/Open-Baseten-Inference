"""A lightweight fake OpenAI-compatible backend used for integration tests and the fake engine.

Run as ``python -m servepilot.testing.fake_openai --port 30001 ...``. It implements:

* ``GET /health``, ``GET /v1/models``
* ``POST /v1/chat/completions`` and ``POST /v1/completions`` (streaming and non-streaming)
* ``GET /_fake/stats`` and ``POST /_fake/config`` for tests (inflight counters, runtime tuning)
* ``POST /_fake/crash`` to simulate a dying backend

Latency is simulated: time-to-first-token and per-token delays grow linearly once the number of
in-flight generations exceeds ``--capacity``, which yields realistic throughput plateaus for the
concurrency sweep. Startup failure modes (``--startup-mode oom|crash|hang|unsupported``) emit
realistic engine log lines so failure classification can be exercised end to end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

OOM_LOG = """INFO 09-04 12:00:01 [model_runner.py:1108] Starting to load model ...
INFO 09-04 12:00:09 [model_runner.py:1117] Loading model weights took 61.02 GB
Traceback (most recent call last):
  File "/opt/engine/worker.py", line 233, in determine_num_available_blocks
    self.model_runner.profile_run()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB. GPU 0 has a total capacity of 79.10 GiB of which 1.20 GiB is free.
"""
CRASH_LOG = """INFO 09-04 12:00:01 [engine.py:44] Initializing engine
Traceback (most recent call last):
  File "/opt/engine/engine.py", line 88, in <module>
    main()
RuntimeError: worker process exited unexpectedly (simulated crash)
"""
UNSUPPORTED_LOG = """INFO 09-04 12:00:01 [config.py:52] Resolving model architecture
ValueError: Model architectures ['FooBarForCausalLM'] are not supported for now.
"""
NCCL_LOG = """INFO 09-04 12:00:01 [parallel_state.py:1004] world_size=2 rank=0
torch.distributed.DistBackendError: NCCL error in: ProcessGroupNCCL.cpp:1970, unhandled system error (run with NCCL_DEBUG=INFO for details)
ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error.
"""


@dataclass
class FakeConfig:
    model: str = "fake-model"
    ttft_ms: float = 20.0
    tpot_ms: float = 4.0
    capacity: int = 32
    error_rate: float = 0.0
    max_model_len: int = 8192
    default_max_tokens: int = 64
    seed: int = 0
    # Per-token delay multiplier grows as (inflight / capacity) beyond capacity.
    overload_slope: float = 1.0


@dataclass
class FakeState:
    config: FakeConfig
    inflight: int = 0
    running: int = 0
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    peak_inflight: int = 0
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    admission: asyncio.Semaphore | None = None

    def semaphore(self) -> asyncio.Semaphore:
        # Created lazily so it binds to the running event loop.
        if self.admission is None or self._cap != self.config.capacity:
            self.admission = asyncio.Semaphore(max(1, self.config.capacity))
            self._cap = self.config.capacity
        return self.admission

    _cap: int = -1


def _count_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _prompt_tokens_from_body(body: dict[str, Any]) -> int:
    if "messages" in body and isinstance(body["messages"], list):
        total = 0
        for m in body["messages"]:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, str):
                total += _count_tokens(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        total += _count_tokens(part["text"])
        return total + 4 * len(body["messages"])
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = " ".join(str(p) for p in prompt)
    return _count_tokens(str(prompt))


def create_app(config: FakeConfig) -> FastAPI:
    app = FastAPI(title="fake-openai")
    state = FakeState(config=config, rng=random.Random(config.seed))
    app.state.fake = state

    def load_factor() -> float:
        # Per-token time grows mildly with the running batch, like a real decode step.
        cap = max(1, state.config.capacity)
        return 1.0 + state.config.overload_slope * 0.5 * (state.running / cap)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": state.config.model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "fake",
                    "max_model_len": state.config.max_model_len,
                }
            ],
        }

    @app.get("/_fake/stats")
    async def stats() -> dict[str, Any]:
        return {
            "inflight": state.inflight,
            "running": state.running,
            "total_requests": state.total_requests,
            "completed_requests": state.completed_requests,
            "failed_requests": state.failed_requests,
            "peak_inflight": state.peak_inflight,
            "pid": os.getpid(),
            "config": state.config.__dict__,
        }

    @app.post("/_fake/config")
    async def configure(request: Request) -> dict[str, Any]:
        body = await request.json()
        for key, value in body.items():
            if hasattr(state.config, key):
                setattr(state.config, key, type(getattr(state.config, key))(value))
        return {"config": state.config.__dict__}

    @app.post("/_fake/crash")
    async def crash() -> JSONResponse:
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, os._exit, 3)
        return JSONResponse({"crashing": True})

    async def generate(body: dict[str, Any], *, chat: bool) -> Any:
        state.total_requests += 1
        state.inflight += 1
        state.peak_inflight = max(state.peak_inflight, state.inflight)
        handed_to_stream = False
        try:
            if state.config.error_rate > 0 and state.rng.random() < state.config.error_rate:
                state.failed_requests += 1
                return JSONResponse(
                    {"error": {"message": "simulated backend failure", "type": "server_error"}},
                    status_code=500,
                )
            prompt_tokens = _prompt_tokens_from_body(body)
            requested = (
                body.get("max_tokens")
                or body.get("max_completion_tokens")
                or state.config.default_max_tokens
            )
            n_tokens = max(1, min(int(requested), state.config.max_model_len - prompt_tokens))
            stream = bool(body.get("stream", False))
            include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
            model_name = body.get("model") or state.config.model
            request_id = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            if stream:
                handed_to_stream = True  # the generator releases the in-flight slot
                return StreamingResponse(
                    _stream_tokens(
                        request_id=request_id,
                        created=created,
                        model_name=model_name,
                        n_tokens=n_tokens,
                        prompt_tokens=prompt_tokens,
                        chat=chat,
                        include_usage=include_usage,
                        state=state,
                        load_factor=load_factor,
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Fake-Backend": str(os.getpid())},
                )

            async with state.semaphore():
                state.running += 1
                try:
                    await asyncio.sleep(state.config.ttft_ms / 1000.0 * load_factor())
                    per_token = state.config.tpot_ms / 1000.0
                    for _ in range(n_tokens - 1):
                        await asyncio.sleep(per_token * load_factor())
                finally:
                    state.running -= 1
            text = " ".join(f"tok{i}" for i in range(n_tokens))
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": n_tokens,
                "total_tokens": prompt_tokens + n_tokens,
            }
            if chat:
                payload = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": usage,
                }
            else:
                payload = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {"index": 0, "text": text, "finish_reason": "length", "logprobs": None}
                    ],
                    "usage": usage,
                }
            state.completed_requests += 1
            return JSONResponse(payload, headers={"X-Fake-Backend": str(os.getpid())})
        finally:
            if not handed_to_stream:
                state.inflight -= 1

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        return await generate(body, chat=True)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Any:
        body = await request.json()
        return await generate(body, chat=False)

    return app


async def _stream_tokens(
    *,
    request_id: str,
    created: int,
    model_name: str,
    n_tokens: int,
    prompt_tokens: int,
    chat: bool,
    include_usage: bool,
    state: FakeState,
    load_factor: Any,
) -> AsyncIterator[bytes]:
    """SSE generator; the in-flight counter is released when the stream ends (or the client leaves)."""
    sem = state.semaphore()
    admitted = False
    try:
        await sem.acquire()
        admitted = True
        state.running += 1
        await asyncio.sleep(state.config.ttft_ms / 1000.0 * load_factor())
        per_token = state.config.tpot_ms / 1000.0
        for i in range(n_tokens):
            if i > 0:
                await asyncio.sleep(per_token * load_factor())
            piece = ("" if i == 0 else " ") + f"tok{i}"
            finish = "length" if i == n_tokens - 1 else None
            if chat:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": piece}
                            if i == 0
                            else {"content": piece},
                            "finish_reason": finish,
                        }
                    ],
                }
            else:
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {"index": 0, "text": piece, "finish_reason": finish, "logprobs": None}
                    ],
                }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        if include_usage:
            usage_chunk: dict[str, Any] = {
                "id": request_id,
                "object": "chat.completion.chunk" if chat else "text_completion",
                "created": created,
                "model": model_name,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": n_tokens,
                    "total_tokens": prompt_tokens + n_tokens,
                },
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        state.completed_requests += 1
    finally:
        if admitted:
            state.running -= 1
            sem.release()
        state.inflight -= 1


def _simulate_startup_failure(mode: str, delay: float) -> None:
    time.sleep(delay)
    if mode == "oom":
        sys.stderr.write(OOM_LOG)
        sys.stderr.flush()
        sys.exit(1)
    if mode == "crash":
        sys.stderr.write(CRASH_LOG)
        sys.stderr.flush()
        sys.exit(2)
    if mode == "unsupported":
        sys.stderr.write(UNSUPPORTED_LOG)
        sys.stderr.flush()
        sys.exit(1)
    if mode == "nccl":
        sys.stderr.write(NCCL_LOG)
        sys.stderr.flush()
        sys.exit(1)
    if mode == "hang":
        while True:
            time.sleep(3600)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fake OpenAI-compatible backend for ServePilot tests"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default="fake-model")
    parser.add_argument("--ttft-ms", type=float, default=20.0)
    parser.add_argument("--tpot-ms", type=float, default=4.0)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--error-rate", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--default-max-tokens", type=int, default=64)
    parser.add_argument("--startup-delay", type=float, default=0.0)
    parser.add_argument(
        "--startup-mode",
        choices=["ok", "oom", "crash", "hang", "unsupported", "nccl"],
        default="ok",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    sys.stdout.write(
        f"INFO fake-openai starting model={args.model} port={args.port} pid={os.getpid()}\n"
    )
    sys.stdout.flush()
    if args.startup_mode != "ok":
        _simulate_startup_failure(args.startup_mode, args.startup_delay)
        return
    if args.startup_delay > 0:
        time.sleep(args.startup_delay)

    import uvicorn

    config = FakeConfig(
        model=args.model,
        ttft_ms=args.ttft_ms,
        tpot_ms=args.tpot_ms,
        capacity=args.capacity,
        error_rate=args.error_rate,
        max_model_len=args.max_model_len,
        default_max_tokens=args.default_max_tokens,
        seed=args.seed,
    )
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
