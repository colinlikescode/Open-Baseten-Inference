"""Cloud catalog, SkyPilot task rendering, the `sky` wrapper (fake binary) and the cloud CLI."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from servepilot.cli.app import app
from servepilot.cloud.catalog import (
    CATALOG,
    GPU_MODELS,
    catalog_snapshot,
    find_instance,
    list_shapes,
    parse_accelerators,
)
from servepilot.cloud.skypilot import (
    RAY_PORT,
    CloudError,
    LaunchRequest,
    SkyClient,
    render_task,
    render_task_yaml,
    wait_for_endpoint,
)
from servepilot.constants import GIB
from servepilot.exceptions import ConfigurationError
from tests.conftest import make_local_model

runner = CliRunner()


class TestCatalog:
    def test_every_shape_points_at_a_known_gpu(self) -> None:
        for shapes in CATALOG.values():
            for s in shapes:
                assert s.gpu in GPU_MODELS
                assert s.gpu_count >= 1

    def test_find_and_parse(self) -> None:
        shape = find_instance("AWS", "p5.48xlarge")
        assert shape.gpu == "H100" and shape.gpu_count == 8 and shape.nvlink
        model, count = parse_accelerators("h100:8")
        assert model.name.startswith("NVIDIA H100") and count == 8
        assert parse_accelerators("L4")[1] == 1
        with pytest.raises(ConfigurationError):
            find_instance("aws", "not-a-type")
        with pytest.raises(ConfigurationError):
            find_instance("mars", "x")
        with pytest.raises(ConfigurationError):
            parse_accelerators("H100:zero")
        with pytest.raises(ConfigurationError):
            parse_accelerators("Q9000:2")
        with pytest.raises(ConfigurationError):
            catalog_snapshot(provider="aws")

    def test_single_node_snapshot(self) -> None:
        snap = catalog_snapshot(provider="gcp", instance="a3-highgpu-8g")
        assert snap.gpu_count == 8 and not snap.is_cluster and snap.provider == "catalog"
        assert snap.topology.summary() == "full NVLink mesh"
        assert snap.gpus[0].total_memory_bytes == 81559 * 1024 * 1024
        pcie = catalog_snapshot(provider="aws", instance="g6e.12xlarge")
        assert not pcie.topology.has_nvlink() and pcie.gpus[0].compute_capability == "8.9"
        one = catalog_snapshot(provider="gcp", accelerators="A100-80GB")
        assert one.gpu_count == 1

    def test_multi_node_snapshot(self) -> None:
        snap = catalog_snapshot(provider="azure", instance="Standard_ND96isr_H100_v5", nodes=3)
        assert snap.is_cluster and len(snap.nodes) == 3 and snap.gpu_count == 24
        assert snap.gpu(9).node_id == "node-1" and snap.gpu(9).device_index_on_node == 1
        with pytest.raises(ConfigurationError):
            catalog_snapshot(provider="aws", instance="p5.48xlarge", nodes=0)

    def test_list(self) -> None:
        assert len(list_shapes("gcp")) == len(CATALOG["gcp"])
        assert len(list_shapes()) == sum(len(v) for v in CATALOG.values())
        with pytest.raises(ConfigurationError):
            list_shapes("nope")


class TestTaskRendering:
    def test_single_node_instance(self) -> None:
        req = LaunchRequest(
            model="Qwen/Qwen3-32B",
            cloud="aws",
            instance="p5.48xlarge",
            region="us-east-1",
            serve_args=["--objective", "latency"],
            hf_token="hf_secret",
        )
        task = render_task(req)
        assert req.name == "servepilot-qwen3-32b"
        assert task["num_nodes"] == 1
        assert task["resources"] == {
            "infra": "aws/us-east-1",
            "ports": 8000,
            "use_spot": False,
            "disk_size": 512,
            "instance_type": "p5.48xlarge",
        }
        assert task["envs"]["SERVEPILOT_MODEL"] == "Qwen/Qwen3-32B"
        assert task["envs"]["SERVEPILOT_ARGS"] == "--objective latency"
        assert task["envs"]["SERVEPILOT_ENGINE_PACKAGES"].startswith("vllm")
        assert task["secrets"] == {"HF_TOKEN": "hf_secret"}
        assert "servepilot-venv" in task["setup"] and 'serve "$SERVEPILOT_MODEL"' in task["run"]
        # `servepilot==1.0.0[ray]` is not a valid requirement; Ray is installed separately and
        # version-matched to the engine's so every client can join the same cluster.
        assert "[ray]" not in task["setup"] and "RAY_SPEC" in task["setup"]
        redacted = render_task_yaml(req, redact=True)
        assert "hf_secret" not in redacted and "HF_TOKEN: '***'" in redacted
        parsed = yaml.safe_load(render_task_yaml(req))
        assert parsed["secrets"]["HF_TOKEN"] == "hf_secret" and parsed["run"] == task["run"]

    def test_scripts_are_valid_shell(self) -> None:
        import shutil
        import subprocess

        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash not available")
        task = render_task(LaunchRequest(model="m", cloud="aws", accelerators="H100:8", nodes=2))
        for script in (task["setup"], task["run"]):
            check = subprocess.run([bash, "-n"], input=script, capture_output=True, text=True)
            assert check.returncode == 0, check.stderr

    def test_multi_node_accelerators_spot_autostop_sglang(self) -> None:
        req = LaunchRequest(
            model="deepseek-ai/DeepSeek-V3",
            cloud="GCP",
            accelerators="H200:8",
            nodes=2,
            spot=True,
            autostop_minutes=30,
            engine="sglang",
            name="dsv3",
        )
        task = render_task(req)
        assert (
            task["resources"]["accelerators"] == "H200:8" and task["resources"]["use_spot"] is True
        )
        assert task["resources"]["autostop"] == {"idle_minutes": 30, "down": True}
        assert (
            task["envs"]["SERVEPILOT_ENGINE"] == "sglang"
            and task["envs"]["SERVEPILOT_ENGINE_PACKAGES"] == "sglang[all]"
        )
        # Expanded unquoted by the setup script: literal quote characters would reach pip.
        assert '"' not in task["envs"]["SERVEPILOT_ENGINE_PACKAGES"]
        assert task["envs"]["SERVEPILOT_ARGS"] == "--engine sglang"
        assert f"--port={RAY_PORT}" in task["run"] and "SKYPILOT_NODE_RANK" in task["run"]
        assert "secrets" not in task

    def test_validation(self) -> None:
        with pytest.raises(ConfigurationError):
            LaunchRequest(model="m", cloud="aws")
        with pytest.raises(ConfigurationError):
            LaunchRequest(model="m", cloud="digitalocean-mars", instance="x")
        # Only AWS, GCP and Azure are supported; other SkyPilot clouds are rejected.
        for other in ("lambda", "runpod", "kubernetes"):
            with pytest.raises(ConfigurationError, match="unsupported cloud"):
                LaunchRequest(model="m", cloud=other, accelerators="H100:8")
        assert LaunchRequest(model="m", cloud="Google", accelerators="H100:8").cloud == "gcp"
        with pytest.raises(ConfigurationError):
            LaunchRequest(model="m", cloud="aws", instance="x", nodes=0)
        with pytest.raises(ConfigurationError):
            LaunchRequest(model="m", cloud="aws", instance="x", engine="trt")
        with pytest.raises(ConfigurationError):
            LaunchRequest(model="m", cloud="aws", instance="x", name="Bad Name")


FAKE_SKY = """#!/usr/bin/env bash
# Records every call and answers like the real `sky` CLI would.
echo "$@" >> "$FAKE_SKY_LOG"
case "$1" in
  launch) echo "Launching cluster $3 ..."; echo "Job submitted"; exit ${FAKE_SKY_LAUNCH_EXIT:-0} ;;
  status)
    if [ "$3" = "--endpoint" ]; then echo "http://203.0.113.10:$4"; exit 0; fi
    if [ "$2" = "-o" ] || [ "$3" = "-o" ]; then echo '[{"name": "servepilot-demo", "status": "UP", "infra": "aws (us-east-1)", "resources_str": "1x p5.48xlarge", "launched_at": 1}]'; exit 0; fi
    echo "NAME  STATUS"; exit 0 ;;
  down) echo "Terminating $2"; exit 0 ;;
  logs) echo "log line"; exit 0 ;;
  check) echo "AWS: enabled"; exit 0 ;;
esac
exit 1
"""


@pytest.fixture
def fake_sky(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "sky"
    script.write_text(FAKE_SKY)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "sky.log"
    monkeypatch.setenv("SERVEPILOT_SKY_BIN", str(script))
    monkeypatch.setenv("FAKE_SKY_LOG", str(log))
    return log


class TestSkyClient:
    def test_missing_binary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SERVEPILOT_SKY_BIN", "/definitely/missing/sky")
        client = SkyClient()
        assert not client.available()
        with pytest.raises(CloudError) as exc:
            client.require()
        assert 'pip install -e ".[cloud]"' in exc.value.render()

    def test_operations(self, fake_sky: Path, tmp_path: Path) -> None:
        seen: list[str] = []
        client = SkyClient(on_output=seen.append)
        task = tmp_path / "task.yaml"
        task.write_text("name: x\n")
        client.launch(task, "servepilot-demo")
        assert client.endpoint("servepilot-demo", 8000) == "http://203.0.113.10:8000"
        entries = client.status()
        assert entries[0]["name"] == "servepilot-demo" and entries[0]["status"] == "UP"
        client.down("servepilot-demo")
        assert "log line" in client.logs("servepilot-demo").stdout
        assert client.check().returncode == 0
        calls = fake_sky.read_text().splitlines()
        assert calls[0] == f"launch -c servepilot-demo -y -d {task}"
        assert "status servepilot-demo --endpoint 8000" in calls
        assert "down servepilot-demo -y" in calls
        assert any(line.startswith("$ ") for line in seen)

    def test_launch_failure(
        self, fake_sky: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_SKY_LAUNCH_EXIT", "3")
        task = tmp_path / "task.yaml"
        task.write_text("name: x\n")
        with pytest.raises(CloudError) as exc:
            SkyClient().launch(task, "c")
        assert "sky check" in exc.value.render()
        assert "sky status c --refresh" in exc.value.render()
        assert "reuse --name c" in exc.value.render()

    def test_wait_for_endpoint(self, fake_sky: Path) -> None:
        client = SkyClient()
        probes: list[str] = []

        def probe(url: str) -> bool:
            probes.append(url)
            return len(probes) >= 2

        url = wait_for_endpoint(
            client, "c", 8000, timeout_seconds=30, poll_seconds=0.01, probe=probe
        )
        assert url == "http://203.0.113.10:8000" and len(probes) == 2
        assert (
            wait_for_endpoint(
                client, "c", 8000, timeout_seconds=0.05, poll_seconds=0.01, probe=lambda u: False
            )
            is None
        )

    def test_endpoint_address_does_not_imply_readiness(self, fake_sky: Path) -> None:
        messages: list[str] = []
        assert (
            wait_for_endpoint(
                SkyClient(),
                "c",
                8000,
                timeout_seconds=0.03,
                poll_seconds=0.01,
                probe=lambda url: False,
                on_progress=messages.append,
            )
            is None
        )
        assert messages and all("not ready yet" in message for message in messages)


@pytest.fixture
def model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SERVEPILOT_VLLM_PYTHON", "/nonexistent/python")
    monkeypatch.setenv("SERVEPILOT_SGLANG_PYTHON", "/nonexistent/python")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return make_local_model(tmp_path, "llama3_8b", 16 * GIB)


class TestCloudCLI:
    def test_plan_with_cloud_shape_needs_no_local_gpu_or_engine(self, model_dir: Path) -> None:
        result = runner.invoke(
            app,
            [
                "plan",
                str(model_dir),
                "--cloud",
                "aws",
                "--instance",
                "p5.48xlarge",
                "--nodes",
                "2",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert (
            payload["hardware"]["gpu_count"] == 16 and payload["hardware"]["provider"] == "catalog"
        )
        engines = {c["engine"] for c in payload["planning"]["candidates"]}
        assert engines == {"vllm", "sglang"}
        only = runner.invoke(
            app,
            [
                "plan",
                str(model_dir),
                "--cloud",
                "gcp",
                "--accelerators",
                "L4:4",
                "--engine",
                "vllm",
                "--json",
            ],
        )
        assert {c["engine"] for c in json.loads(only.stdout)["planning"]["candidates"]} == {"vllm"}
        bad = runner.invoke(app, ["plan", str(model_dir), "--cloud", "aws", "--instance", "nope"])
        assert bad.exit_code == 2

    def test_inspect_cloud(self) -> None:
        result = runner.invoke(app, ["inspect", "cloud", "azure", "--json"])
        payload = json.loads(result.stdout)
        assert any(p["instance"] == "Standard_ND96isr_H100_v5" for p in payload)
        assert {
            p["provider"]
            for p in json.loads(runner.invoke(app, ["inspect", "cloud", "--json"]).stdout)
        } == {"aws", "gcp", "azure"}
        human = runner.invoke(app, ["inspect", "cloud"])
        assert human.exit_code == 0 and "p5.48xlarge" in human.stdout

    def test_launch_dry_run(self, model_dir: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "launch",
                str(model_dir),
                "--cloud",
                "aws",
                "--instance",
                "p5.48xlarge",
                "--nodes",
                "2",
                "--objective",
                "latency",
                "--expected-concurrency",
                "16",
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["name"] == "servepilot-llama3-8b" and payload["command"].startswith(
            "sky launch -c servepilot-llama3-8b -y -d "
        )
        task = yaml.safe_load(payload["task"])
        assert task["num_nodes"] == 2 and task["resources"]["instance_type"] == "p5.48xlarge"
        assert task["envs"]["SERVEPILOT_ARGS"] == "--objective latency --expected-concurrency 16"
        assert (
            Path(payload["task_path"]).exists()
            and oct(Path(payload["task_path"]).stat().st_mode & 0o777) == "0o600"
        )
        # Only vLLM gets installed for the default engine, so the pre-launch plan shows only vLLM.
        engines = {c["engine"] for c in payload["planning"]["candidates"]}
        assert engines == {"vllm"} and task["envs"]["SERVEPILOT_ENGINE"] == "vllm"

    def test_launch_without_sky_is_actionable(
        self, model_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SERVEPILOT_SKY_BIN", "/definitely/missing/sky")
        result = runner.invoke(
            app, ["launch", str(model_dir), "--cloud", "aws", "--instance", "p5.48xlarge"]
        )
        assert result.exit_code == 3 and "sky check" in result.output

    def test_launch_down_clusters_with_fake_sky(
        self, model_dir: Path, fake_sky: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "servepilot.cli.cloud_cmd.wait_for_endpoint", lambda *a, **k: "http://203.0.113.10:8000"
        )
        result = runner.invoke(
            app,
            [
                "launch",
                str(model_dir),
                "--cloud",
                "gcp",
                "--instance",
                "a3-highgpu-8g",
                "--name",
                "servepilot-demo",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert (
            payload["endpoint"] == "http://203.0.113.10:8000"
            and payload["down"] == "servepilot down servepilot-demo"
        )
        calls = fake_sky.read_text()
        assert "launch -c servepilot-demo -y -d" in calls
        listed = runner.invoke(app, ["clusters", "--json"])
        assert json.loads(listed.stdout)[0]["name"] == "servepilot-demo"
        human = runner.invoke(app, ["clusters"])
        assert "servepilot-demo" in human.stdout
        down = runner.invoke(app, ["down", "servepilot-demo", "--json"])
        assert json.loads(down.stdout) == {"name": "servepilot-demo", "down": True}
        assert "down servepilot-demo -y" in fake_sky.read_text()

    def test_launch_no_wait_with_accelerators_still_launches(
        self, model_dir: Path, fake_sky: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "launch",
                str(model_dir),
                "--cloud",
                "azure",
                "--accelerators",
                "H100:2",
                "--no-wait",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["endpoint"] is None
        # No catalog shape named, but --accelerators still plans and launches.
        assert "launch -c servepilot-llama3-8b -y -d" in fake_sky.read_text()

    def test_launch_rejects_other_clouds(self, model_dir: Path, fake_sky: Path) -> None:
        result = runner.invoke(
            app, ["launch", str(model_dir), "--cloud", "lambda", "--accelerators", "H100:8"]
        )
        assert result.exit_code == 2 and "aws, gcp, azure" in result.output
        assert not fake_sky.exists()  # `sky` was never called

    def test_hf_token_goes_into_secrets_only(
        self, model_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_abcdefghijklmnop")
        result = runner.invoke(
            app,
            [
                "launch",
                str(model_dir),
                "--cloud",
                "aws",
                "--instance",
                "p5.48xlarge",
                "--dry-run",
                "--json",
            ],
        )
        payload = json.loads(result.stdout)
        assert "hf_abcdefghijklmnop" not in payload["task"] and "HF_TOKEN: '***'" in payload["task"]
        assert "hf_abcdefghijklmnop" in Path(payload["task_path"]).read_text()
        assert "hf_abcdefghijklmnop" not in result.stderr if hasattr(result, "stderr") else True
        os.environ.pop("HF_TOKEN", None)
