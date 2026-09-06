# Real GPU validation

Validated on 6 September 2026 using one GCP Spot `a4-highgpu-8g` VM in
`europe-west1-b`: eight NVIDIA B200 GPUs, a full NVLink mesh, and NVIDIA driver
580.159.04. These are measurements from real engine processes and generated tokens.

## Qwen3-32B throughput

The full default tuning run used vLLM 0.28.0 and `Qwen/Qwen3-32B` in bfloat16,
revision `9216db5781bf21249d130ec9da846c4624c16137`. The chat workload used an 8,192-token
context, prompt lengths of 512/2,048 tokens at p50/p95, and output lengths of 256/768.
The synthetic workload uses a fixed seed; these results describe that workload.

The initial comparison held concurrency at 256 and memory fraction at 0.91:

| GPUs per replica | Replicas | Output tokens/second | Failed requests |
| --- | --- | ---: | ---: |
| 1 | 8 | 7,867 | 0 |
| 2 | 4 | 9,688 | 0 |
| 4 | 2 | 10,551 | 0 |
| 8 | 1 | 9,695 | 0 |

After concurrency and memory tuning, final confirmation selected **two four-GPU replicas**,
memory fraction **0.95**, and total concurrency **1,024**. It completed **2,048 of 2,048
requests** at **16,236 output tokens/second** over 44.76 seconds. The p95 time to first token
was 7.41 seconds and p95 request latency was 24.79 seconds. This operating point optimizes
throughput; latency-sensitive traffic needs a latency objective and its expected concurrency.

Across all 27 measurements, 21,568 requests completed with zero failed client requests.
Some high-concurrency trials needed a retry after a backend TCP connection timeout.

## Qwen3-32B latency and Ray

A second run used an isolated, single-node Ray cluster with the same eight GPUs. It compared
vLLM 0.28.0 and SGLang 0.5.19 with the latency objective, expected concurrency 16, and a
per-replica running-request cap of 64. All six structural candidates succeeded: TP2, TP4,
and TP8 with each engine. There were no failed requests across 384 measured requests.

The winner was **SGLang with one eight-GPU replica**, at concurrency 16. Final confirmation
completed 128/128 requests with p95 first-token latency **107.9 ms**, p95 request latency
**3.76 seconds**, p95 time per output token **5.20 ms**, and **2,949 output tokens/second**.
This used real Ray process actors, NVML discovery, and engine processes on one physical node.

The tuning command, once both engines and the Ray cluster were available, was:

```bash
servepilot tune Qwen/Qwen3-32B --engine auto --objective latency \
  --expected-concurrency 16 --max-concurrency 64 --ray-address RAY_ADDRESS --json
```

## Serving checks

The selected plan was started as a serving deployment and checked through its external GCP
endpoint using both the OpenAI Python client and raw HTTP:

- Model listing, chat completions, text completions, and streaming with usage counts.
- SSE completion markers and backend error responses passed through the proxy.
- Concurrent traffic reached both replicas; cancelled streams released all routing slots.
- Health, status, and Prometheus metrics remained available.
- Streaming and non-streaming CLI benchmarks each completed 32 of 32 requests.
- Killing one backend process left the other replica serving. All 27 recovery-probe requests
  succeeded, and the supervisor restarted the killed replica automatically.
- Stopping the deployment released allocations on all eight GPUs.

All six tests in the real GPU suite passed. They also exercised vLLM 0.28.0 and SGLang 0.5.19 with
`Qwen/Qwen2.5-0.5B-Instruct`, including streaming, completions, status, stop, and memory cleanup.
Ray hardware discovery identified all eight physical GPUs, and the Linux Ray integration
tests passed on an isolated Ray instance alongside SkyPilot's own runtime.

The final Linux regression suite passed **329 tests**, with the six GPU tests run separately.
Real dry runs reused the saved throughput winner when compatible and honored engine, topology,
memory, and concurrency overrides without allocating GPU memory. Lint, type checking, and
source/wheel builds also passed.

A final cloud relaunch uploaded the updated checkout, reused the saved throughput plan,
and passed the external API checks again. Hashes of the changed deployed modules matched
the local source; ServePilot and vLLM both used Ray 2.58.0.

After validation, the test VM, its persistent disk, and its cluster firewall rule were deleted.

## Reproduction

With GCP credentials and quota configured, install this checkout and launch:

```bash
pip install -e ".[cloud]"
sky check gcp
servepilot launch Qwen/Qwen3-32B --cloud gcp --accelerators B200:8 \
  --spot --autostop 20 --name servepilot-validation
```

Cloud launches build and upload the current checkout. Record the commit, model revision,
engine versions, and tuning JSON when comparing another run. Instance availability, engine
versions, workload, and cache state can change the measurements. Remove the rented cluster
when finished:

```bash
servepilot down servepilot-validation
```
