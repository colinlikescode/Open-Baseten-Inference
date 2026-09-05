# Open-Baseten-Inference

Finds the fastest way to serve an LLM on NVIDIA GPUs, then serves it, in your own cloud account.

**What it is.** You bring a GCP, Azure or AWS account (1-8 GPUs single node, or a Ray
cluster). ServePilot benchmarks the real serving layouts (replicas, GPUs per replica, vLLM or
SGLang, load) for time to first token and throughput, picks the winner, and puts it behind one
OpenAI-compatible URL. SkyPilot creates the machines in your account; ServePilot serves on them
and tears them down. GCP, Azure and AWS are the only clouds supported.

**Why it exists.** Picking a serving layout by hand is guesswork: vLLM or SGLang, how many GPUs
per copy of the model, how many copies. The obvious choice (one copy across every GPU) often loses
to two smaller copies by 30%, and no spec sheet will tell you which. ServePilot measures instead
of guessing, on your GPUs, in your account.

```bash
pip install servepilot        # plus: pip install vllm   and/or   pip install "sglang[all]"
servepilot serve Qwen/Qwen3-32B
```

API is at `http://127.0.0.1:8000/v1`. Ctrl-C stops everything and frees the GPUs.

```bash
servepilot serve Qwen/Qwen3-32B --objective latency --expected-concurrency 32
servepilot serve Qwen/Qwen3-32B --profile long-context --gpus 0,1,2,3 --engine sglang

# in your own GCP / Azure / AWS account (billed to you). Credentials must already be set up:
# gcloud auth, az login, or aws configure. `sky check` confirms SkyPilot can see them.
pip install "servepilot[cloud]" && sky check
servepilot launch Qwen/Qwen3-32B --cloud gcp --accelerators H100:8
servepilot launch Qwen/Qwen3-235B-A22B --cloud aws --instance p5.48xlarge --nodes 2
servepilot down servepilot-qwen3-32b

# what would fit on a machine you have not rented yet (no GPUs or credentials needed)
servepilot plan Qwen/Qwen3-235B-A22B --cloud azure --instance Standard_ND96isr_H100_v5 --nodes 2
```

## Commands

```text
servepilot serve MODEL                 tune if needed, then serve        --retune --no-tune --dry-run
servepilot plan | tune MODEL           show layouts / benchmark them and save the winner
servepilot launch MODEL --cloud gcp|azure|aws --instance TYPE | --accelerators H100:8 [--nodes N]
servepilot clusters | down NAME        list / tear down launched machines
servepilot benchmark URL               benchmark any OpenAI-compatible server
servepilot doctor | status | stop      check the machine / what is running / stop it
servepilot inspect hardware|model|cloud
```

Every command takes `--json`. `-v` shows progress, `-vv` shows engine logs.

## Docs

[how it works](docs/architecture.md) · [planner](docs/planner.md) ·
[benchmarking](docs/benchmarking.md) · [engines](docs/engines.md) ·
[config file](servepilot.example.yaml) · [security](SECURITY.md)

Needs Linux, NVIDIA drivers, Python 3.11+. Tests and planning run anywhere:
`pip install -e ".[dev]" && pytest`. Apache 2.0.
