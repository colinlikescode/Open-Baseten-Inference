"""Async OpenAI-compatible benchmark client.

Measures TTFT (first SSE event carrying model output), end-to-end latency and TPOT per request.
Token counts come from the API ``usage`` object when present, otherwise from the tokenizer.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from servepilot.benchmark.workload import BenchmarkRequest
from servepilot.models.tokenizer import TokenCounter
from servepilot.schemas.benchmark import BenchmarkEndpoint, RequestBenchmarkResult


def build_payload(
    request: BenchmarkRequest,
    *,
    model: str,
    endpoint: BenchmarkEndpoint,
    stream: bool,
    ignore_eos: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": request.max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }
    if ignore_eos:
        # vLLM/SGLang extension: force the full output length so measured output tokens match
        # the workload profile instead of depending on where the model chooses to stop.
        payload["ignore_eos"] = True
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if endpoint == "chat":
        payload["messages"] = [{"role": "user", "content": request.prompt}]
    else:
        payload["prompt"] = request.prompt
    return payload


def _extract_text(chunk: dict[str, Any], endpoint: BenchmarkEndpoint) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    if endpoint == "chat":
        delta = first.get("delta") or {}
        content = delta.get("content")
        if content is None and isinstance(first.get("message"), dict):
            content = first["message"].get("content")
        return content or ""
    return first.get("text") or ""


class BenchmarkClient:
    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        tokenizer: TokenCounter,
        timeout_seconds: float,
        api_key: str | None = None,
        max_connections: int = 2048,
        ignore_eos: bool = True,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._tok = tokenizer
        self._ignore_eos = ignore_eos
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=30.0),
            headers=headers,
            limits=httpx.Limits(
                max_connections=max_connections, max_keepalive_connections=max_connections
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> BenchmarkClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _path(self, endpoint: BenchmarkEndpoint) -> str:
        return "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"

    async def run_request(
        self, request: BenchmarkRequest, *, endpoint: BenchmarkEndpoint, stream: bool
    ) -> RequestBenchmarkResult:
        payload = build_payload(
            request,
            model=self._model,
            endpoint=endpoint,
            stream=stream,
            ignore_eos=self._ignore_eos,
        )
        url = self._base + self._path(endpoint)
        started = time.perf_counter()
        if stream:
            return await self._run_streaming(url, payload, request, endpoint, started)
        return await self._run_blocking(url, payload, request, endpoint, started)

    async def _run_blocking(
        self,
        url: str,
        payload: dict[str, Any],
        request: BenchmarkRequest,
        endpoint: BenchmarkEndpoint,
        started: float,
    ) -> RequestBenchmarkResult:
        try:
            resp = await self._client.post(url, json=payload)
        except httpx.HTTPError as exc:
            return RequestBenchmarkResult(
                success=False,
                input_tokens=request.input_tokens,
                started_at=started,
                error=f"{type(exc).__name__}: {exc}",
            )
        completed = time.perf_counter()
        if resp.status_code >= 400:
            return RequestBenchmarkResult(
                success=False,
                input_tokens=request.input_tokens,
                started_at=started,
                completed_at=completed,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
            )
        try:
            body = resp.json()
        except ValueError:
            return RequestBenchmarkResult(
                success=False,
                input_tokens=request.input_tokens,
                started_at=started,
                completed_at=completed,
                error="invalid JSON response",
                status_code=resp.status_code,
            )
        usage = body.get("usage") or {}
        text = _extract_text(body, endpoint)
        output_tokens = int(usage.get("completion_tokens") or 0) or self._tok.count(text)
        input_tokens = int(usage.get("prompt_tokens") or 0) or request.input_tokens
        e2e_ms = (completed - started) * 1000.0
        tpot = (e2e_ms / output_tokens) if output_tokens > 0 else None
        return RequestBenchmarkResult(
            success=output_tokens > 0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            started_at=started,
            completed_at=completed,
            e2e_latency_ms=e2e_ms,
            tpot_ms=tpot,
            status_code=resp.status_code,
            error=None if output_tokens > 0 else "no output tokens",
        )

    async def _run_streaming(
        self,
        url: str,
        payload: dict[str, Any],
        request: BenchmarkRequest,
        endpoint: BenchmarkEndpoint,
        started: float,
    ) -> RequestBenchmarkResult:
        first_token_at: float | None = None
        chunks_with_text = 0
        text_parts: list[str] = []
        usage: dict[str, Any] = {}
        status_code: int | None = None
        try:
            async with self._client.stream("POST", url, json=payload) as resp:
                status_code = resp.status_code
                if resp.status_code >= 400:
                    body = await resp.aread()
                    return RequestBenchmarkResult(
                        success=False,
                        input_tokens=request.input_tokens,
                        started_at=started,
                        completed_at=time.perf_counter(),
                        error=f"HTTP {resp.status_code}: {body[:200]!r}",
                        status_code=resp.status_code,
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    text = _extract_text(chunk, endpoint)
                    if text:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        chunks_with_text += 1
                        text_parts.append(text)
        except httpx.HTTPError as exc:
            return RequestBenchmarkResult(
                success=False,
                input_tokens=request.input_tokens,
                started_at=started,
                first_token_at=first_token_at,
                completed_at=time.perf_counter(),
                error=f"{type(exc).__name__}: {exc}",
                status_code=status_code,
            )
        completed = time.perf_counter()
        output_tokens = int(usage.get("completion_tokens") or 0)
        if output_tokens <= 0:
            output_tokens = (
                self._tok.count("".join(text_parts)) if self._tok.exact else chunks_with_text
            )
        input_tokens = int(usage.get("prompt_tokens") or 0) or request.input_tokens
        e2e_ms = (completed - started) * 1000.0
        ttft_ms = (first_token_at - started) * 1000.0 if first_token_at is not None else None
        tpot_ms: float | None = None
        if ttft_ms is not None and output_tokens > 1:
            tpot_ms = (e2e_ms - ttft_ms) / (output_tokens - 1)
        return RequestBenchmarkResult(
            success=output_tokens > 0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            started_at=started,
            first_token_at=first_token_at,
            completed_at=completed,
            ttft_ms=ttft_ms,
            e2e_latency_ms=e2e_ms,
            tpot_ms=tpot_ms,
            status_code=status_code,
            error=None if output_tokens > 0 else "no output tokens",
        )
