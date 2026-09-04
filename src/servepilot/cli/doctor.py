"""Environment checks for ``servepilot doctor``."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from typing import Literal

import httpx

from servepilot.constants import GIB
from servepilot.engines.registry import EngineRegistry
from servepilot.exceptions import HardwareError
from servepilot.hardware.base import HardwareProvider
from servepilot.runtime.ports import port_is_free
from servepilot.settings import ServePilotSettings

Status = Literal["ok", "warn", "fail", "info"]


@dataclass
class Check:
    section: str
    status: Status
    label: str
    detail: str = ""
    hints: list[str] = field(default_factory=list)


def _python_check() -> Check:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    return Check(
        "System",
        "ok" if ok else "fail",
        f"Python {v.major}.{v.minor}.{v.micro}",
        "" if ok else "ServePilot requires Python 3.11+",
    )


def _platform_check() -> Check:
    system = platform.system()
    if system == "Linux":
        return Check("System", "ok", f"Linux {platform.machine()}")
    return Check(
        "System",
        "warn",
        f"{system} {platform.machine()}",
        "GPU serving requires Linux; planning, tests and the fake engine work anywhere.",
    )


def _disk_check(path: str) -> Check:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return Check("System", "warn", "disk space", f"could not read disk usage for {path}: {exc}")
    free_gib = usage.free / GIB
    if free_gib < 20:
        return Check(
            "System",
            "warn",
            f"{free_gib:.0f} GiB free on {path}",
            "model downloads need tens to hundreds of GiB",
            ["Free disk space or set HF_HOME to a larger volume."],
        )
    return Check("System", "ok", f"{free_gib:.0f} GiB free on {path}")


def hardware_checks(provider: HardwareProvider) -> list[Check]:
    checks: list[Check] = []
    try:
        snap = provider.snapshot()
    except HardwareError as exc:
        checks.append(Check("NVIDIA", "fail", "NVML unavailable", exc.message, exc.hints))
        return checks
    checks.append(
        Check(
            "NVIDIA",
            "ok",
            "NVML available" if snap.provider == "nvml" else f"{snap.provider} provider",
        )
    )
    checks.append(
        Check(
            "NVIDIA",
            "ok" if snap.driver_version else "warn",
            f"driver {snap.driver_version or 'unknown'}",
        )
    )
    checks.append(
        Check(
            "NVIDIA",
            "ok" if snap.cuda_version else "warn",
            f"CUDA {snap.cuda_version or 'unknown'}",
        )
    )
    if snap.gpu_count == 0:
        checks.append(
            Check(
                "NVIDIA",
                "fail",
                "no GPUs visible",
                "",
                ["Run nvidia-smi; check CUDA_VISIBLE_DEVICES."],
            )
        )
    else:
        names = sorted({g.name for g in snap.gpus})
        checks.append(Check("NVIDIA", "ok", f"{snap.gpu_count} GPU(s) visible: {', '.join(names)}"))
        if not snap.is_homogeneous:
            checks.append(
                Check(
                    "NVIDIA", "warn", "mixed GPU types", "select a homogeneous subset with --gpus"
                )
            )
        busy = [g for g in snap.gpus if g.used_fraction > 0.05]
        if busy:
            checks.append(
                Check(
                    "NVIDIA",
                    "warn",
                    f"{len(busy)} GPU(s) already have memory allocated",
                    ", ".join(
                        f"GPU {g.index}: {g.used_memory_bytes / GIB:.1f} GiB used" for g in busy
                    ),
                )
            )
    if snap.gpu_count > 1:
        if snap.topology.available:
            checks.append(
                Check("NVIDIA", "ok", f"GPU topology readable ({snap.topology.summary()})")
            )
        else:
            checks.append(
                Check(
                    "NVIDIA",
                    "warn",
                    "GPU topology unavailable",
                    snap.topology.error or "",
                    ["Grouping falls back to index order."],
                )
            )
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        checks.append(
            Check(
                "NVIDIA",
                "info",
                f"CUDA_VISIBLE_DEVICES={cvd}",
                "ServePilot only plans against the visible devices",
            )
        )
    return checks


def engine_checks(registry: EngineRegistry) -> list[Check]:
    checks: list[Check] = []
    for engine in registry.all():
        if engine.engine_name.value == "fake":
            checks.append(Check("Inference engines", "info", "fake engine enabled (testing only)"))
            continue
        if engine.is_available():
            python = getattr(engine, "python", lambda: None)()
            detail = f"via {python}" if python and python != sys.executable else ""
            checks.append(
                Check("Inference engines", "ok", f"{engine.name()} {engine.version()}", detail)
            )
        else:
            checks.append(
                Check(
                    "Inference engines",
                    "warn",
                    f"{engine.name()} not found",
                    "",
                    engine.unavailable_hints(),
                )
            )
    if not registry.available():
        checks.append(
            Check(
                "Inference engines",
                "fail",
                "no inference engine available",
                "",
                ['pip install vllm  or  pip install "sglang[all]"'],
            )
        )
    return checks


def torch_check() -> list[Check]:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return [
            Check(
                "PyTorch",
                "info",
                "PyTorch not installed in this environment",
                "engines bring their own",
            )
        ]
    checks = [Check("PyTorch", "ok", f"torch {torch.__version__}")]
    cuda = getattr(torch.version, "cuda", None)
    checks.append(Check("PyTorch", "ok" if cuda else "warn", f"torch CUDA {cuda or 'unavailable'}"))
    return checks


def huggingface_checks(timeout: float = 5.0) -> list[Check]:
    checks: list[Check] = []
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        checks.append(
            Check("Hugging Face", "info", "HF_HUB_OFFLINE=1: only cached/local models are usable")
        )
    else:
        try:
            resp = httpx.get(
                f"{endpoint}/api/models?limit=1", timeout=timeout, follow_redirects=True
            )
            ok = resp.status_code < 500
            checks.append(
                Check(
                    "Hugging Face",
                    "ok" if ok else "warn",
                    f"{endpoint} reachable" if ok else f"{endpoint} returned {resp.status_code}",
                )
            )
        except httpx.HTTPError as exc:
            checks.append(
                Check(
                    "Hugging Face",
                    "warn",
                    f"{endpoint} unreachable",
                    f"{type(exc).__name__}",
                    ["Gated/private models need network access; local model paths work offline."],
                )
            )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            token = None
    checks.append(
        Check(
            "Hugging Face",
            "ok" if token else "info",
            "HF token configured" if token else "no HF token (needed for gated models)",
        )
    )
    return checks


def port_check(port: int) -> Check:
    if port_is_free(port):
        return Check("Network", "ok", f"port {port} is free")
    return Check(
        "Network",
        "warn",
        f"port {port} is in use",
        "",
        [f"Pass --port to `servepilot serve` or stop the process using {port}."],
    )


def run_all(
    provider: HardwareProvider,
    registry: EngineRegistry,
    settings: ServePilotSettings,
    *,
    public_port: int,
    hf_timeout: float = 5.0,
) -> list[Check]:
    checks = [
        _platform_check(),
        _python_check(),
        _disk_check(os.environ.get("HF_HOME") or str(settings.cache_dir.parent)),
    ]
    checks.extend(hardware_checks(provider))
    checks.extend(engine_checks(registry))
    checks.extend(torch_check())
    checks.extend(huggingface_checks(hf_timeout))
    checks.append(port_check(public_port))
    return checks


def overall_status(checks: list[Check]) -> Status:
    if any(c.status == "fail" for c in checks):
        return "fail"
    if any(c.status == "warn" for c in checks):
        return "warn"
    return "ok"
