# Planner

The planner turns hardware + model + workload into a short list of layouts worth
benchmarking. It only rules things out. It never declares a winner.

## Static estimates

For each GPU, the engine's budget is `memory_fraction × total memory`. ServePilot derives
the fraction from what is actually **free**, minus a safety reserve
(`max(8% of total, 1 GiB)`, `--memory-headroom` changes the 8%), rounded down to two
decimals and capped at 0.95. Memory used by other processes is never counted as available.

The budget has to cover:

| Part | How it is estimated |
| --- | --- |
| weights | measured bytes ÷ shards, +1% for uneven sharding. Bytes come from local files, Hub file sizes, safetensors metadata, or a config-based estimate, in that order, and the source is always shown. |
| KV cache | `2 × layers × kv_heads × head_dim × bytes` per token, split across TP ranks (heads are replicated when there are fewer heads than ranks). MLA models use `layers × (kv_lora_rank + rope_dim) × bytes`, replicated. Unknown architectures (Mamba, Jamba, ...) report "unknown" and need at least 2 GiB spare instead. |
| activations | 2.5% of device memory |
| CUDA graphs | 1 GiB |
| communication | 256 MiB per rank when TP > 1 |
| engine overhead | 1.5 GiB (CUDA context, Python runtime) |

A layout is "estimated viable" when at least one full-context sequence fits in the remaining
KV cache. The estimated concurrency shown in `servepilot plan` is remaining KV cache ÷ the
p50 sequence length.

These numbers are in `constants.py`. They are deliberately conservative; engines profile the
real values at launch, and the tuner measures the truth.

## Candidates

For `G` GPUs per machine:

1. Tensor-parallel sizes are the divisors of `G` that also divide the model's attention
   head count.
2. For each TP, the replica count is what fills the GPUs: `G ÷ TP` per machine.
3. GPUs are grouped by topology. Pairs with NVLink score highest, then closer PCIe ancestry.
   Machines with up to 8 GPUs are searched exhaustively; bigger ones use a greedy pass. Ties
   break on lowest index, so results are stable. Groups never span machines unless the plan
   is a vLLM Ray layout for a model too big for one machine.
4. Each engine says whether it supports the model and the layout (`SupportResult` with
   reasons). MoE models add expert-parallel variants where the engine supports them, and a
   DP-attention variant (attention data parallel, experts sharded) for the architectures each
   engine is known to run it on: SGLang via `--enable-dp-attention`, vLLM 0.9+ via
   `--data-parallel-size N --enable-expert-parallel`.
5. The memory model marks each candidate viable or not.
6. Duplicates are removed. Candidates are ranked: smaller TP first for throughput (more
   copies, no communication), larger TP first for latency.

`--tp`, `--replicas`, `--gpus`, `--engine`, `--context-length`, `--max-concurrency`,
`--memory-fraction` narrow the search. They never skip validation: an impossible combination
fails with a clear message. The context is capped at the model's maximum; when even the
workload's p95 prompt + output would not fit, that is an error too (force it with
`--context-length N --allow-context-override`).

Mixed GPU models in one machine stop the planner with a list of homogeneous subsets you can
pick with `--gpus`.

## Clusters

With a Ray cluster, candidates are built per machine (TP ≤ GPUs per node, copies across all
nodes). If nothing fits inside one machine, vLLM Ray layouts are added: TP inside each node
and pipeline parallel across nodes, plus a single TP group spanning nodes when head counts
allow it.

## Scoring

Documented with the formulas in [benchmarking.md](benchmarking.md).

## Explanations

`servepilot plan` prints why the smallest TP is what it is and which layouts will be
evaluated. After tuning, the rationale is built from the recorded results only: measured
throughput or latency of every layout at its best operating point, failed launches with
their classified reason, where the concurrency sweep peaked, why the plateau rule chose the
operating point, whether the memory fraction change helped, and the final confirmation
numbers.
