"""Failure classification from engine logs and exit codes."""

from __future__ import annotations

import pytest

from servepilot.engines.failures import classify_failure, summarize_error, tail
from servepilot.schemas.benchmark import FailureType
from servepilot.testing.fake_openai import CRASH_LOG, NCCL_LOG, OOM_LOG, UNSUPPORTED_LOG


@pytest.mark.parametrize(
    ("stderr", "code", "expected"),
    [
        (OOM_LOG, 1, FailureType.OOM),
        (
            "ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization",
            1,
            FailureType.OOM,
        ),
        (
            "RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.",
            1,
            FailureType.OOM,
        ),
        ("RuntimeError: CUDA error: device-side assert triggered", 1, FailureType.CUDA_ERROR),
        ("RuntimeError: Found no NVIDIA driver on your system.", 1, FailureType.CUDA_ERROR),
        (NCCL_LOG, 1, FailureType.NCCL_ERROR),
        (UNSUPPORTED_LOG, 1, FailureType.MODEL_UNSUPPORTED),
        (
            "ValueError: Total number of attention heads (30) must be divisible by tensor parallel size (4).",
            1,
            FailureType.MODEL_UNSUPPORTED,
        ),
        ("api_server.py: error: unrecognized arguments: --bogus", 2, FailureType.INVALID_ARGUMENT),
        ("OSError: [Errno 98] Address already in use", 1, FailureType.PORT_CONFLICT),
        (
            "huggingface_hub.errors.GatedRepoError: 401 Client Error. Access to model meta-llama/x is restricted",
            1,
            FailureType.AUTH_ERROR,
        ),
        (
            "requests.exceptions.ConnectionError: Max retries exceeded with url: /api/models/foo",
            1,
            FailureType.MODEL_DOWNLOAD_ERROR,
        ),
        (CRASH_LOG, 2, FailureType.ENGINE_CRASH),
        ("", 137, FailureType.ENGINE_CRASH),
    ],
)
def test_classification(stderr: str, code: int, expected: FailureType) -> None:
    failure = classify_failure(code, "", stderr, engine="vllm")
    assert failure.type == expected
    assert failure.exit_code == code
    assert failure.message.startswith("vllm")


def test_timeout_and_unknown() -> None:
    timeout = classify_failure(None, "INFO loading weights...", "", timed_out=True)
    assert timeout.type == FailureType.STARTUP_TIMEOUT
    unknown = classify_failure(0, "", "")
    assert unknown.type == FailureType.UNKNOWN
    still_unknown = classify_failure(None, "", "")
    assert still_unknown.type == FailureType.UNKNOWN


def test_summary_and_tail() -> None:
    assert summarize_error(OOM_LOG) is not None and summarize_error(OOM_LOG).startswith(
        "torch.OutOfMemoryError"
    )  # type: ignore[union-attr]
    assert summarize_error("nothing here") is None
    long = "\n".join(f"line {i}" for i in range(500))
    assert tail(long, 3) == "line 497\nline 498\nline 499"
    failure = classify_failure(1, "", long)
    assert failure.stderr_tail.count("\n") < 100


def test_sigkill_note() -> None:
    failure = classify_failure(-9, "", "")
    assert failure.type == FailureType.ENGINE_CRASH and "SIGKILL" in failure.message
