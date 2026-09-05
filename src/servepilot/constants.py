"""Centralized constants for ServePilot.

Anything that is a tunable heuristic lives here so it can be discovered, documented and
overridden in one place rather than being scattered as magic numbers through the code base.
"""

from __future__ import annotations

from enum import IntEnum

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
KIB = 1024
MIB = 1024 * KIB
GIB = 1024 * MIB

# ---------------------------------------------------------------------------
# Exit codes (documented in docs/architecture.md)
# ---------------------------------------------------------------------------


class ExitCode(IntEnum):
    """Process exit codes used by the CLI."""

    SUCCESS = 0
    CONFIGURATION_ERROR = 2
    ENVIRONMENT_ERROR = 3
    MODEL_ERROR = 4
    ENGINE_UNAVAILABLE = 5
    NO_VIABLE_PLAN = 6
    RUNTIME_FAILURE = 7
    BENCHMARK_ERROR = 8
    CACHE_ERROR = 9
    INTERRUPTED = 130
    UNEXPECTED_ERROR = 1


# ---------------------------------------------------------------------------
# Schema / persistence
# ---------------------------------------------------------------------------
TUNING_RECORD_SCHEMA_VERSION = 1
RUNTIME_STATE_SCHEMA_VERSION = 1
CACHE_DIR_NAME = "servepilot"
TUNING_CACHE_SUBDIR = "tuning"
RUNTIME_STATE_FILENAME = "runtime.json"

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------
DEFAULT_PUBLIC_HOST = "127.0.0.1"
DEFAULT_PUBLIC_PORT = 8000
DEFAULT_BACKEND_PORT_START = 30000
DEFAULT_BACKEND_PORT_END = 30999
DEFAULT_STARTUP_TIMEOUT_SECONDS = 1200.0
DEFAULT_READINESS_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_GRACEFUL_SHUTDOWN_SECONDS = 20.0
DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS = 5.0
DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS = 5.0
DEFAULT_UNHEALTHY_AFTER_FAILURES = 3
DEFAULT_REPLICA_MAX_RESTARTS = 3
DEFAULT_REPLICA_RESTART_WINDOW_SECONDS = 600.0
DEFAULT_REPLICA_RESTART_BACKOFF_SECONDS = 10.0
DEFAULT_STARTUP_STAGGER_SECONDS = 2.0
DEFAULT_PROXY_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_QUEUE_DEPTH = 1024

# ---------------------------------------------------------------------------
# Memory model heuristics (see docs/planner.md)
# ---------------------------------------------------------------------------
DEFAULT_MEMORY_HEADROOM_FRACTION = 0.08
MIN_HEADROOM_BYTES = 1 * GIB
# CUDA context + engine Python runtime + allocator fragmentation.
ENGINE_OVERHEAD_BYTES = int(1.5 * GIB)
# CUDA graph capture memory; engines capture a set of batch sizes.
CUDA_GRAPH_BYTES = 1 * GIB
# NCCL/communication buffers per rank when TP > 1.
COMM_BUFFER_BYTES_PER_TP_RANK = 256 * MIB
# Fraction of the device reserved for activations/intermediate tensors (engines profile the real
# value at startup; 2.5% of an 80 GB device is ~2 GiB).
ACTIVATION_FRACTION_OF_TOTAL = 0.025
# Sharded weights are never perfectly balanced (replicated norms/biases); pad the per-rank estimate.
WEIGHT_SHARD_OVERHEAD_FRACTION = 0.01
# Do not let engines claim more than this fraction of total memory.
MAX_MEMORY_FRACTION = 0.95
MIN_MEMORY_FRACTION = 0.30
# A candidate is estimated viable only if the KV cache fits at least this many sequences at
# the p95 workload lengths.
MIN_KV_SEQUENCES = 1
# Warn when other processes already hold more than this fraction of a selected GPU.
BUSY_GPU_USED_FRACTION = 0.05

# ---------------------------------------------------------------------------
# Tuning heuristics (see docs/benchmarking.md)
# ---------------------------------------------------------------------------
DEFAULT_MAX_STRUCTURAL_CANDIDATES = 6
DEFAULT_TOP_K_FOR_CONCURRENCY_SWEEP = 2
DEFAULT_STAGE_A_REQUESTS = 32
DEFAULT_SWEEP_REQUESTS_PER_POINT = 32
DEFAULT_FINAL_CONFIRMATION_MULTIPLIER = 4
DEFAULT_SWEEP_START_CONCURRENCY = 4
DEFAULT_SWEEP_MAX_CONCURRENCY = 1024
DEFAULT_SWEEP_GROWTH_FACTOR = 2
DEFAULT_SWEEP_REFINEMENT_POINTS = 2
# Plateau rule: stop raising concurrency when throughput improves less than this fraction...
PLATEAU_MIN_THROUGHPUT_GAIN = 0.02
# ...while p95 latency rises by more than this fraction.
PLATEAU_LATENCY_RISE_THRESHOLD = 0.10
# Stop sweeping after throughput falls by this fraction from the best point.
SWEEP_THROUGHPUT_DROP_STOP = 0.05
# Candidates with a higher error rate than this are invalid for selection.
MAX_ACCEPTABLE_ERROR_RATE = 0.01
# Scoring penalty applied per unit of error rate below the hard threshold.
ERROR_RATE_PENALTY_WEIGHT = 5.0
# Memory tuning step for stage C.
MEMORY_TUNING_STEP = 0.04
MEMORY_TUNING_MIN_GAIN = 0.02
# Warmup requests scale modestly with concurrency.
WARMUP_MIN_REQUESTS = 2
WARMUP_MAX_REQUESTS = 16
# Benchmark request timeout.
DEFAULT_BENCHMARK_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_BENCHMARK_SEED = 1234
# GPU sampling interval during benchmarks.
GPU_SAMPLE_INTERVAL_SECONDS = 0.5

# ---------------------------------------------------------------------------
# Logging / process capture
# ---------------------------------------------------------------------------
PROCESS_LOG_TAIL_LINES = 400
PROCESS_LOG_TAIL_BYTES = 256 * KIB
FAILURE_TAIL_LINES = 60
