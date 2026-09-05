"""Ray-backed hardware discovery and remote process launching on a local Ray instance.

These tests run wherever ``ray`` is installed (``pip install "servepilot[ray]"``); without it they
are skipped. A single-node Ray instance exercises the exact actor/placement code path used on
multi-node clusters.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from servepilot.constants import GIB
from servepilot.engines.base import LaunchSpec
from servepilot.hardware.fingerprint import hardware_fingerprint
from servepilot.models.inspector import inspect_local_model
from servepilot.planner.candidates import generate_candidates
from servepilot.runtime.ports import PortAllocator
from servepilot.runtime.replicas import ReplicaSet
from servepilot.runtime.router import ReplicaRouter
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.workload import WorkloadProfile
from servepilot.testing import fake_hardware as fh
from servepilot.testing.fake_engine import FakeEngine, FakeEngineBehavior
from tests.conftest import make_local_model

ray = pytest.importorskip("ray")
pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def ray_cluster() -> Iterator[str]:
    ray.init(
        num_cpus=4,
        num_gpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    try:
        yield "auto"
    finally:
        ray.shutdown()


def _fake_node_snapshot(node_id: str, node_ip: str) -> HardwareSnapshot:
    return fh.h100x2()


def test_ray_hardware_provider_merges_nodes(ray_cluster: str) -> None:
    from servepilot.cluster.ray_provider import RayHardwareProvider

    provider = RayHardwareProvider(ray_cluster, node_snapshot_fn=_fake_node_snapshot)
    snap = provider.snapshot()
    assert snap.provider == "ray" and snap.gpu_count == 2 and len(snap.nodes) == 1
    node = snap.nodes[0]
    assert node.node_id and node.node_ip and snap.gpu(0).node_id == node.node_id
    assert hardware_fingerprint(snap) == hardware_fingerprint(provider.snapshot())


async def test_ray_launcher_runs_fake_engine_and_terminates(ray_cluster: str) -> None:
    import os
    import sys

    from servepilot.cluster.ray_launcher import RayLauncher
    from servepilot.cluster.ray_provider import RayHardwareProvider

    node = (
        RayHardwareProvider(ray_cluster, node_snapshot_fn=_fake_node_snapshot).snapshot().nodes[0]
    )
    port = PortAllocator(35000, 35999).allocate()
    # Same minimal environment the fake engine builds; PYTHONPATH keeps a source checkout
    # importable inside the remote process.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", "/"),
        "PYTHONUNBUFFERED": "1",
    }
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    spec = LaunchSpec(
        executable=sys.executable,
        args=[
            "-m",
            "servepilot.testing.fake_openai",
            "--port",
            str(port),
            "--ttft-ms",
            "1",
            "--tpot-ms",
            "1",
        ],
        env=env,
        host="127.0.0.1",
        port=port,
        gpu_ids=[0],
        redacted_display_command="fake",
        replica_id="ray-r0",
        node_id=node.node_id,
        node_ip="127.0.0.1",
    )
    launcher = RayLauncher(ray_cluster)
    handle = await launcher.launch(spec)
    try:
        assert (
            handle.is_running() and handle.pid is not None and launcher.supports_node(node.node_id)
        )
        async with httpx.AsyncClient(timeout=5) as client:
            for _ in range(200):
                try:
                    if (await client.get(f"{spec.base_url}/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    await asyncio.sleep(0.1)
            else:
                raise AssertionError(
                    f"remote fake engine never became ready: {handle.stdout_tail()} {handle.stderr_tail()}"
                )
            resp = await client.post(
                f"{spec.base_url}/v1/completions",
                json={"model": "x", "prompt": "hi", "max_tokens": 2},
            )
            assert resp.status_code == 200
        await handle.refresh()  # the background poller mirrors logs once per second
        assert "fake-openai starting" in handle.stdout_tail()
    finally:
        await launcher.shutdown_all()
    assert not handle.is_running() and launcher.tracked() == []
    async with httpx.AsyncClient(timeout=2) as client:
        with pytest.raises(httpx.HTTPError):
            await client.get(f"{spec.base_url}/health")


async def test_replica_set_through_ray_launcher(ray_cluster: str, tmp_path: Path) -> None:
    from servepilot.cluster.ray_launcher import RayLauncher
    from servepilot.cluster.ray_provider import RayHardwareProvider

    snapshot = RayHardwareProvider(ray_cluster, node_snapshot_fn=_fake_node_snapshot).snapshot()
    for g in snapshot.gpus:
        g.node_ip = "127.0.0.1"
    for n in snapshot.nodes:
        n.node_ip = "127.0.0.1"
    model = inspect_local_model(make_local_model(tmp_path, "llama3_8b", 16 * GIB))
    workload = WorkloadProfile(
        name="tiny",
        input_tokens_p50=16,
        input_tokens_p95=32,
        output_tokens_p50=8,
        output_tokens_p95=16,
        max_context_tokens=1024,
    )
    engine = FakeEngine(FakeEngineBehavior(base_tpot_ms=1.0, base_ttft_ms=1.0))
    planning = generate_candidates(snapshot, model, workload, [engine])
    plan = next(p for p in planning.candidates if p.tensor_parallel_size == 1)
    assert plan.replica_count == 2 and plan.replica_nodes is not None
    router = ReplicaRouter()
    launcher = RayLauncher(ray_cluster)
    replica_set = ReplicaSet(
        plan=plan,
        model=model,
        engine=engine,
        launcher=launcher,
        ports=PortAllocator(36000, 36999),
        hardware=snapshot,
        router=router,
        startup_timeout=60,
        stagger_seconds=0,
    )
    try:
        await replica_set.start()
        assert len(router.healthy_replicas()) == 2
        async with httpx.AsyncClient(timeout=10) as client:
            for replica in replica_set.replicas:
                assert (await client.get(f"{replica.base_url}/v1/models")).status_code == 200
    finally:
        assert await replica_set.stop()
        await launcher.shutdown_all()
    assert router.replicas == []
