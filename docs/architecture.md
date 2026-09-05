# How ServePilot works

```text
Hardware + Model + Workload
        │
        ▼
  Static planner      which layouts can fit (TP, copies, GPU groups, engine)
        │
        ▼
  Candidate plans     impossible ones dropped, each with a reason
        │
        ▼
  Empirical tuner     launch, warm up, benchmark, compare, next
        │
        ▼
  Measured winner     saved with every number behind the decision
        │
        ▼
  Runtime             engine replicas + router + OpenAI-compatible API
```

The planner proposes. The benchmark decides. The runtime executes. Those three never share
code paths.

## Pieces

| Package | Job | Talks to |
| --- | --- | --- |
| `hardware/` | Read GPUs, memory, NVLink/PCIe topology through NVML. Fingerprint the machine. | pynvml |
| `models/` | Read `config.json` and repository metadata. Size weights. Estimate KV cache per token. Never loads weights. | Hugging Face Hub, local files |
| `planner/` | Memory model, candidate generation, GPU grouping, pruning, scoring, plateau rule, Pareto front, explanations. | nothing external |
| `engines/` | vLLM and SGLang adapters: launch commands, readiness, failure classification. Process supervision with process groups. | the engine processes |
| `benchmark/` | Deterministic prompt generation, async OpenAI client, metrics, GPU sampling, staged tuner. | the router in front of a candidate |
| `runtime/` | Replica set, health checks with bounded restarts, least-in-flight router, ports, runtime state file. | engine processes |
| `api/` | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/status`, `/metrics`. Streaming pass-through proxy. | clients |
| `cache/` | Tuning records keyed by hardware + model + workload fingerprints. Atomic writes. | disk |
| `cluster/` | Ray: cluster-wide hardware snapshot, engine processes placed on nodes through actors. | Ray |
| `cloud/` | Instance shape catalog for planning; SkyPilot task generation and the `sky` CLI for launching. | SkyPilot |
| `testing/` | Fake hardware fixtures, fake engine, fake OpenAI server. Used by the test suite and by you, without a GPU. | - |

## Abstractions worth knowing

- `HardwareProvider.snapshot()` is the only way anything learns about GPUs. NVML, Ray and
  the fake provider all implement it.
- `InferenceEngine` is the engine interface: `supports`, `build_launch_spec`,
  `wait_until_ready`, `classify_failure`. All engine flags live in the adapters.
- `Launcher` starts processes locally or on Ray nodes. Both return a `ProcessHandle` with the
  same methods, so the tuner and health checker do not care where a replica runs.
- `CandidateEvaluator.open(plan)` launches a candidate and returns a session you can
  benchmark. The production one launches real replicas plus the real router. Tests use a
  scripted one.

## Serving path

1. `ReplicaSet` starts one engine process per replica (first alone so the model is cached
   once, then the rest staggered), waits for `/v1/models` on each, registers them with the
   router.
2. `ReplicaRouter` hands each request to the healthy replica with the fewest in-flight
   requests. A semaphore caps concurrency at the tuned value; a bounded queue waits behind it
   and returns 503 when full.
3. The proxy forwards bytes as they arrive. A request is retried once on another replica
   only if the first replica failed before sending any response. The process runs on uvloop
   with uvicorn's httptools parser; the benchmark load generator shares that loop, so this
   also raises the ceiling on what tuning can measure.
   Cancellation and client disconnects release the concurrency slot, including failures before
   response headers are sent. A backend disconnect after partial output aborts the response so
   clients can detect the incomplete generation.
4. `HealthChecker` probes replicas, pulls dead or failing ones out of rotation, restarts them
   with backoff up to a limit, and keeps the others serving.
5. A runtime state file (`~/.local/state/servepilot/runtime.json`) records PIDs, ports and
   process start times so `servepilot status` and `servepilot stop` work from another shell
   and never signal an unrelated process.

## Cleanup

Every engine process runs in its own process group. Shutdown sends SIGTERM to the group,
waits, then SIGKILLs anything left. An `atexit` hook does the same if ServePilot itself dies.
After a benchmark, ServePilot waits for GPU memory to return near the baseline before it
launches the next candidate.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | success |
| 1 | unexpected error (re-run with `-vv` for the traceback) |
| 2 | invalid flags or config file |
| 3 | environment: no NVML/GPUs, mixed GPU types, missing `sky` |
| 4 | model could not be inspected (missing, gated, unreadable) |
| 5 | no usable inference engine |
| 6 | no layout fits, or every candidate failed |
| 7 | a launch or runtime failure (engine did not start, deployment already running) |
| 8 | a benchmark could not run |
| 9 | tuning cache unreadable or from another schema version |
| 130 | interrupted with Ctrl-C |
