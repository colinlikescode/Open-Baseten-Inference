"""Instance shapes for planning before you rent anything.

The catalog only holds what NVML would report on the machine: GPU model, count per node,
memory per GPU, compute capability and whether the GPUs are NVLinked. The planner treats a
catalog snapshot exactly like a real one. Nothing here changes how ServePilot behaves on real
hardware; it just lets ``servepilot plan --cloud ...`` answer "will this fit?" ahead of time.

Memory sizes are the values ``nvidia-smi`` shows for each card, in MiB.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from servepilot.exceptions import ConfigurationError
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.testing.fake_hardware import cluster as _cluster_snapshot
from servepilot.testing.fake_hardware import (
    full_nvlink_topology,
    make_gpu,
    make_snapshot,
    pcie_only_topology,
)

MIB = 1024 * 1024


@dataclass(frozen=True)
class GPUModel:
    name: str
    memory_mib: int
    compute_capability: tuple[int, int]
    aliases: tuple[str, ...] = ()

    @property
    def memory_bytes(self) -> int:
        return self.memory_mib * MIB


GPU_MODELS: dict[str, GPUModel] = {
    "H100": GPUModel(
        "NVIDIA H100 80GB HBM3", 81559, (9, 0), ("H100-80GB", "H100-SXM", "H100-PCIE")
    ),
    "H200": GPUModel("NVIDIA H200", 143771, (9, 0), ("H200-141GB",)),
    "GH200": GPUModel("NVIDIA GH200 480GB", 97871, (9, 0)),
    "B200": GPUModel("NVIDIA B200", 183359, (10, 0)),
    "A100-80GB": GPUModel("NVIDIA A100-SXM4-80GB", 81920, (8, 0), ("A100-80",)),
    "A100": GPUModel("NVIDIA A100-SXM4-40GB", 40960, (8, 0), ("A100-40GB",)),
    "L40S": GPUModel("NVIDIA L40S", 46068, (8, 9)),
    "L4": GPUModel("NVIDIA L4", 23034, (8, 9)),
    "A10G": GPUModel("NVIDIA A10G", 23028, (8, 6)),
    "A10": GPUModel("NVIDIA A10", 23028, (8, 6)),
    "V100": GPUModel("Tesla V100-SXM2-16GB", 16384, (7, 0)),
    "T4": GPUModel("Tesla T4", 15360, (7, 5)),
    "RTX6000-ADA": GPUModel("NVIDIA RTX 6000 Ada Generation", 49140, (8, 9), ("RTX6000Ada",)),
    "RTX4090": GPUModel("NVIDIA GeForce RTX 4090", 24564, (8, 9)),
}


@dataclass(frozen=True)
class InstanceShape:
    provider: str
    name: str
    gpu: str
    gpu_count: int
    nvlink: bool
    note: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def gpu_model(self) -> GPUModel:
        return GPU_MODELS[self.gpu]


def _shapes(provider: str, rows: list[tuple[str, str, int, bool]]) -> list[InstanceShape]:
    return [InstanceShape(provider, name, gpu, count, nvlink) for name, gpu, count, nvlink in rows]


CATALOG: dict[str, list[InstanceShape]] = {
    "aws": _shapes(
        "aws",
        [
            ("p6-b200.48xlarge", "B200", 8, True),
            ("p5en.48xlarge", "H200", 8, True),
            ("p5e.48xlarge", "H200", 8, True),
            ("p5.48xlarge", "H100", 8, True),
            ("p4de.24xlarge", "A100-80GB", 8, True),
            ("p4d.24xlarge", "A100", 8, True),
            ("p3.16xlarge", "V100", 8, True),
            ("p3.8xlarge", "V100", 4, True),
            ("p3.2xlarge", "V100", 1, False),
            ("g6e.48xlarge", "L40S", 8, False),
            ("g6e.12xlarge", "L40S", 4, False),
            ("g6e.xlarge", "L40S", 1, False),
            ("g6.48xlarge", "L4", 8, False),
            ("g6.12xlarge", "L4", 4, False),
            ("g6.xlarge", "L4", 1, False),
            ("g5.48xlarge", "A10G", 8, False),
            ("g5.12xlarge", "A10G", 4, False),
            ("g5.xlarge", "A10G", 1, False),
            ("g4dn.12xlarge", "T4", 4, False),
            ("g4dn.xlarge", "T4", 1, False),
        ],
    ),
    "gcp": _shapes(
        "gcp",
        [
            ("a4-highgpu-8g", "B200", 8, True),
            ("a3-ultragpu-8g", "H200", 8, True),
            ("a3-megagpu-8g", "H100", 8, True),
            ("a3-highgpu-8g", "H100", 8, True),
            ("a3-highgpu-4g", "H100", 4, True),
            ("a3-highgpu-2g", "H100", 2, True),
            ("a3-highgpu-1g", "H100", 1, False),
            ("a2-ultragpu-8g", "A100-80GB", 8, True),
            ("a2-ultragpu-4g", "A100-80GB", 4, True),
            ("a2-ultragpu-2g", "A100-80GB", 2, True),
            ("a2-ultragpu-1g", "A100-80GB", 1, False),
            ("a2-highgpu-8g", "A100", 8, True),
            ("a2-highgpu-4g", "A100", 4, True),
            ("a2-highgpu-2g", "A100", 2, True),
            ("a2-highgpu-1g", "A100", 1, False),
            ("g2-standard-96", "L4", 8, False),
            ("g2-standard-48", "L4", 4, False),
            ("g2-standard-24", "L4", 2, False),
            ("g2-standard-4", "L4", 1, False),
        ],
    ),
    "azure": _shapes(
        "azure",
        [
            ("Standard_ND96isr_H200_v5", "H200", 8, True),
            ("Standard_ND96isr_H100_v5", "H100", 8, True),
            ("Standard_ND96amsr_A100_v4", "A100-80GB", 8, True),
            ("Standard_ND96asr_v4", "A100", 8, True),
            ("Standard_NC96ads_A100_v4", "A100-80GB", 4, False),
            ("Standard_NC48ads_A100_v4", "A100-80GB", 2, False),
            ("Standard_NC24ads_A100_v4", "A100-80GB", 1, False),
            ("Standard_NC24s_v3", "V100", 4, False),
            ("Standard_NC6s_v3", "V100", 1, False),
            ("Standard_NV36ads_A10_v5", "A10", 1, False),
        ],
    ),
}

PROVIDERS = sorted(CATALOG)
# The only clouds ServePilot launches on. All three go through SkyPilot, in your own account.
SKYPILOT_PROVIDERS = ("aws", "gcp", "azure")


def normalize_provider(provider: str) -> str:
    p = provider.strip().lower()
    return {
        "google": "gcp",
        "amazon": "aws",
        "microsoft": "azure",
    }.get(p, p)


def find_instance(provider: str, instance: str) -> InstanceShape:
    provider = normalize_provider(provider)
    shapes = CATALOG.get(provider)
    if shapes is None:
        raise ConfigurationError(
            f"no instance catalog for cloud {provider!r} (known: {', '.join(PROVIDERS)}).",
            hints=[
                "Use --accelerators H100:8 (GPU model and count per node) instead of --instance for this cloud."
            ],
        )
    wanted = instance.strip().lower()
    for shape in shapes:
        if shape.name.lower() == wanted or wanted in (a.lower() for a in shape.aliases):
            return shape
    names = ", ".join(s.name for s in shapes)
    raise ConfigurationError(
        f"unknown {provider} instance type {instance!r}.",
        hints=[
            f"Known {provider} shapes: {names}",
            "Or describe the node with --accelerators H100:8.",
        ],
    )


def parse_accelerators(value: str) -> tuple[GPUModel, int]:
    """``H100:8`` → (GPUModel, 8). ``H100`` alone means one GPU."""
    text = value.strip()
    name, _, count_text = text.partition(":")
    count = 1
    if count_text:
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ConfigurationError(
                f"--accelerators count must be an integer, got {value!r}"
            ) from exc
    if count < 1:
        raise ConfigurationError("--accelerators count must be at least 1")
    key = name.strip().upper().replace("_", "-")
    for model_key, model in GPU_MODELS.items():
        if key == model_key.upper() or key in (a.upper() for a in model.aliases):
            return model, count
    raise ConfigurationError(
        f"unknown GPU model {name!r} in --accelerators.",
        hints=[f"Known models: {', '.join(GPU_MODELS)}"],
    )


def catalog_snapshot(
    *,
    provider: str,
    instance: str | None = None,
    accelerators: str | None = None,
    nodes: int = 1,
) -> HardwareSnapshot:
    """Build the hardware snapshot the planner would see on ``nodes`` machines of this shape."""
    if nodes < 1:
        raise ConfigurationError("--nodes must be at least 1")
    provider = normalize_provider(provider)
    if instance:
        shape = find_instance(provider, instance)
        model, count, nvlink = shape.gpu_model, shape.gpu_count, shape.nvlink
        label = f"{provider}/{shape.name}"
    elif accelerators:
        model, count = parse_accelerators(accelerators)
        nvlink = count > 1  # SXM-class multi-GPU nodes are NVLinked on AWS, GCP and Azure
        label = f"{provider}/{accelerators}"
    else:
        raise ConfigurationError(
            "give --instance TYPE or --accelerators GPU:COUNT together with --cloud"
        )

    if nodes == 1:
        gpus = [
            make_gpu(i, model.name, model.memory_bytes, compute_capability=model.compute_capability)
            for i in range(count)
        ]
        idx = list(range(count))
        if count == 1:
            topology = pcie_only_topology(idx)
        else:
            topology = full_nvlink_topology(idx) if nvlink else pcie_only_topology(idx)
        snap = make_snapshot(gpus, topology)
    else:
        snap = _cluster_snapshot(
            nodes, count, name=model.name, total_bytes=model.memory_bytes, nvlink=nvlink
        )
        for g in snap.gpus:
            g.compute_capability_major, g.compute_capability_minor = model.compute_capability
        snap.cluster_address = None
    snap.hostname = label
    snap.provider = "catalog"
    return snap


def list_shapes(provider: str | None = None) -> list[InstanceShape]:
    if provider is None:
        return [s for p in PROVIDERS for s in CATALOG[p]]
    p = normalize_provider(provider)
    if p not in CATALOG:
        raise ConfigurationError(
            f"no instance catalog for cloud {p!r} (known: {', '.join(PROVIDERS)})."
        )
    return list(CATALOG[p])
