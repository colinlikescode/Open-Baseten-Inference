"""Classify engine launch failures from exit codes and log tails."""

from __future__ import annotations

import re

from servepilot.constants import FAILURE_TAIL_LINES
from servepilot.schemas.benchmark import CandidateFailure, FailureType

# Ordered: the first matching category wins. Patterns are matched case-insensitively against
# the combined stdout/stderr tail.
_PATTERNS: list[tuple[FailureType, list[str]]] = [
    (
        FailureType.OOM,
        [
            r"CUDA out of memory",
            r"OutOfMemoryError",
            r"torch\.OutOfMemoryError",
            r"out of memory",
            r"No available memory for the cache blocks",
            r"not enough memory",
            r"Not enough memory\.? Please try to increase --mem-fraction-static",
            r"insufficient memory",
            r"free memory on device .* is less than desired GPU memory utilization",
            r"memory profiling .* exceeds",
            r"cudaErrorMemoryAllocation",
            r"CUBLAS_STATUS_ALLOC_FAILED",
            r"The model's max seq len .* is larger than the maximum number of tokens that can be stored in KV cache",
            r"KV cache is too small",
            r"CUDA_ERROR_OUT_OF_MEMORY",
        ],
    ),
    (
        FailureType.AUTH_ERROR,
        [
            r"401 Client Error",
            r"403 Client Error",
            r"GatedRepoError",
            r"RepositoryNotFoundError",
            r"Access to model .* is restricted",
            r"You must be authenticated",
            r"Invalid user token",
            r"gated repo",
        ],
    ),
    (
        FailureType.MODEL_DOWNLOAD_ERROR,
        [
            r"HfHubHTTPError",
            r"ConnectionError",
            r"Failed to connect to huggingface\.co",
            r"Max retries exceeded",
            r"Read timed out",
            r"No space left on device",
            r"LocalEntryNotFoundError",
            r"RevisionNotFoundError",
            r"EntryNotFoundError",
            r"Cannot find the requested files in the disk cache",
        ],
    ),
    (
        FailureType.PORT_CONFLICT,
        [
            r"address already in use",
            r"\[Errno 98\]",
            r"EADDRINUSE",
            r"port .* is already in use",
            r"Only one usage of each socket address",
        ],
    ),
    (
        FailureType.NCCL_ERROR,
        [
            r"NCCL error",
            r"ncclSystemError",
            r"ncclInternalError",
            r"ncclUnhandledCudaError",
            r"ncclInvalidUsage",
            r"NCCL WARN",
            r"torch\.distributed\.DistBackendError",
            r"Connection reset by peer.*NCCL",
            r"Gloo connectFullMesh failed",
            r"Timed out initializing process group",
        ],
    ),
    (
        FailureType.MODEL_UNSUPPORTED,
        [
            r"Model architectures? \[.*\] (are|is) not supported",
            r"is not supported for now",
            r"Unsupported model type",
            r"not supported by (vLLM|SGLang)",
            r"Cannot find model module",
            r"Unrecognized configuration class",
            r"does not recognize this architecture",
            r"Quantization method .* is not supported",
            r"unsupported quantization",
            r"requires trust_remote_code",
            r"trust_remote_code=True",
            r"The checkpoint you are trying to load has model type .* but Transformers does not recognize",
            r"Total number of attention heads .* must be divisible by tensor parallel size",
            r"is not divisible by tensor parallel size",
            r"Number of experts .* must be divisible",
            r"Bfloat16 is only supported on GPUs with compute capability",
            r"is not supported on this GPU",
        ],
    ),
    (
        FailureType.ENGINE_CRASH,
        [
            # Broken engine environments: missing build tools or Python modules.
            r"No such file or directory: 'ninja'",
            r"ModuleNotFoundError: No module named",
            r"ImportError: (cannot import name|.*\.so)",
            r"undefined symbol",
        ],
    ),
    (
        FailureType.INVALID_ARGUMENT,
        [
            r"error: unrecognized arguments",
            r"error: argument .*: invalid",
            r"error: the following arguments are required",
            r"usage: .*\n.*error",
            r"ValueError: .*(argument|must be|should be|invalid)",
            r"invalid choice",
            r"No such option",
            r"Unknown option",
        ],
    ),
    (
        FailureType.CUDA_ERROR,
        [
            r"CUDA error",
            r"CUDA driver version is insufficient",
            r"no CUDA-capable device is detected",
            r"CUDA_ERROR_",
            r"cudaError",
            r"device-side assert",
            r"an illegal memory access",
            r"RuntimeError: Found no NVIDIA driver",
            r"Torch not compiled with CUDA enabled",
            r"NVML Shared Library Not Found",
            r"Failed to initialize NVML",
            r"cuda runtime error",
            r"cuDNN error",
            r"CUBLAS_STATUS_",
        ],
    ),
]

_COMPILED: list[tuple[FailureType, list[re.Pattern[str]]]] = [
    (ftype, [re.compile(p, re.IGNORECASE) for p in patterns]) for ftype, patterns in _PATTERNS
]

# Exception lines, possibly prefixed by engine log decorations such as
# "(EngineCore pid=123) ERROR 09-04 12:00:00 [core.py:1346] ".
_EXCEPTION_LINE_RE = re.compile(
    r"(\b[A-Za-z_][\w.]*(?:Error|Exception|Interrupt)\b:[^\n]*)$", re.MULTILINE
)
# Wrapper messages that only point at a root cause reported earlier.
_GENERIC_SUMMARIES = (
    "See root cause above",
    "Engine core initialization failed",
    "failed to start",
    "exited unexpectedly",
)


def tail(text: str, lines: int = FAILURE_TAIL_LINES) -> str:
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


def summarize_error(text: str) -> str | None:
    """Return the most informative exception line from a traceback-like log, if any.

    Prefers the last exception that is not a generic wrapper ("see root cause above").
    """
    matches = [str(m).strip() for m in _EXCEPTION_LINE_RE.findall(text)]
    if not matches:
        return None
    specific = [m for m in matches if not any(g in m for g in _GENERIC_SUMMARIES)]
    chosen = specific[-1] if specific else matches[-1]
    return chosen[:400]


def classify_failure(
    exit_code: int | None,
    stdout_tail: str,
    stderr_tail: str,
    *,
    engine: str = "engine",
    timed_out: bool = False,
) -> CandidateFailure:
    """Map logs + exit code to a :class:`CandidateFailure`."""
    combined = f"{stdout_tail}\n{stderr_tail}"
    ftype = FailureType.UNKNOWN
    matched: str | None = None
    for candidate_type, patterns in _COMPILED:
        for pattern in patterns:
            m = pattern.search(combined)
            if m:
                ftype = candidate_type
                matched = m.group(0)
                break
        if matched is not None:
            break

    if ftype == FailureType.UNKNOWN:
        if timed_out:
            ftype = FailureType.STARTUP_TIMEOUT
        elif exit_code is not None and exit_code != 0:
            ftype = FailureType.ENGINE_CRASH
            if exit_code in (-9, 137):
                # SIGKILL usually means the host OOM killer, not GPU OOM; still a crash but note it.
                matched = "process killed with SIGKILL (host OOM killer or external signal)"

    summary = summarize_error(combined) or matched
    message = f"{engine} failed"
    if ftype == FailureType.OOM:
        message = f"{engine} ran out of GPU memory"
    elif ftype == FailureType.STARTUP_TIMEOUT:
        message = f"{engine} did not become ready in time"
    elif ftype == FailureType.MODEL_UNSUPPORTED:
        message = f"{engine} does not support this model/configuration"
    elif ftype == FailureType.INVALID_ARGUMENT:
        message = f"{engine} rejected its launch arguments"
    elif ftype == FailureType.PORT_CONFLICT:
        message = f"{engine} could not bind its port"
    elif ftype == FailureType.NCCL_ERROR:
        message = f"{engine} hit an NCCL/distributed communication error"
    elif ftype == FailureType.CUDA_ERROR:
        message = f"{engine} hit a CUDA error"
    elif ftype == FailureType.MODEL_DOWNLOAD_ERROR:
        message = f"{engine} could not download the model"
    elif ftype == FailureType.AUTH_ERROR:
        message = f"{engine} was denied access to the model repository"
    elif ftype == FailureType.ENGINE_CRASH:
        message = f"{engine} exited unexpectedly (exit code {exit_code})"
    if summary:
        message = f"{message}: {summary}"
    return CandidateFailure(
        type=ftype,
        message=message,
        stderr_tail=tail(stderr_tail),
        stdout_tail=tail(stdout_tail),
        exit_code=exit_code,
    )
