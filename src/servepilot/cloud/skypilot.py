"""Launch ServePilot on AWS, GCP or Azure through SkyPilot.

ServePilot never talks to a cloud API itself. It writes a SkyPilot task (YAML) that:

1. provisions ``nodes`` machines of the requested shape in your AWS, GCP or Azure account,
2. installs ServePilot and an inference engine on each node,
3. on multi-node clusters starts a Ray cluster on port 6379 (SkyPilot keeps its own Ray on
   6380, so we use an explicit address and never ``ray stop``),
4. runs ``servepilot serve MODEL`` on the head node with port 8000 open.

Then it calls the ``sky`` CLI: ``sky launch``, ``sky status --endpoint``, ``sky down``,
``sky logs``. Every command is printed, so you can run the same thing by hand.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from servepilot import __version__
from servepilot.cloud.catalog import SKYPILOT_PROVIDERS, normalize_provider
from servepilot.constants import ExitCode
from servepilot.exceptions import ConfigurationError, ServePilotError
from servepilot.logging import get_logger, redact_secrets

log = get_logger(__name__)

SERVEPILOT_PORT = 8000
RAY_PORT = 6379  # SkyPilot's internal Ray uses 6380; ours must differ.
ENGINE_PACKAGES = {"vllm": "vllm ninja", "sglang": '"sglang[all]"', "auto": "vllm ninja"}


class CloudError(ServePilotError):
    exit_code = ExitCode.ENVIRONMENT_ERROR


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:40] or "model"


@dataclass
class LaunchRequest:
    model: str
    cloud: str
    instance: str | None = None
    accelerators: str | None = None
    nodes: int = 1
    name: str | None = None
    region: str | None = None
    engine: str = "auto"
    spot: bool = False
    autostop_minutes: int | None = None
    disk_size_gb: int = 512
    port: int = SERVEPILOT_PORT
    serve_args: list[str] = field(default_factory=list)
    package: str = f"servepilot=={__version__}"
    hf_token: str | None = None

    def __post_init__(self) -> None:
        self.cloud = normalize_provider(self.cloud)
        if self.cloud not in SKYPILOT_PROVIDERS:
            raise ConfigurationError(
                f"unsupported cloud {self.cloud!r}.",
                hints=[
                    f"ServePilot launches on {', '.join(SKYPILOT_PROVIDERS)} only (via SkyPilot)"
                ],
            )
        if not self.instance and not self.accelerators:
            raise ConfigurationError("give --instance TYPE or --accelerators GPU:COUNT")
        if self.nodes < 1:
            raise ConfigurationError("--nodes must be at least 1")
        if self.engine not in ENGINE_PACKAGES:
            raise ConfigurationError("--engine must be auto, vllm or sglang")
        if self.name is None:
            self.name = f"servepilot-{_slug(self.model.rsplit('/', 1)[-1])}"
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", self.name):
            raise ConfigurationError(
                "--name must be lowercase letters, digits and dashes, starting with a letter"
            )

    @property
    def infra(self) -> str:
        return f"{self.cloud}/{self.region}" if self.region else self.cloud


SETUP_SCRIPT = """set -e
# ServePilot needs Python 3.11+; SkyPilot images do not guarantee that, so we bring uv.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv "$HOME/servepilot-venv" --python 3.12
uv pip install --python "$HOME/servepilot-venv/bin/python" "$SERVEPILOT_PACKAGE[ray]"
# The engine gets its own environment so its CUDA stack never fights with ServePilot's.
uv venv "$HOME/engines/$SERVEPILOT_ENGINE" --python 3.12
uv pip install --python "$HOME/engines/$SERVEPILOT_ENGINE/bin/python" $SERVEPILOT_ENGINE_PACKAGES
"$HOME/servepilot-venv/bin/servepilot" doctor --port $SERVEPILOT_PORT || true
"""

RUN_SCRIPT = """set -e
export PATH="$HOME/.local/bin:$PATH"
SP="$HOME/servepilot-venv/bin/servepilot"
RAY="$HOME/servepilot-venv/bin/ray"
PY="$HOME/servepilot-venv/bin/python"
HEAD_IP=$(echo "$SKYPILOT_NODE_IPS" | head -n1)
RAY_ARGS=""
if [ "${SKYPILOT_NUM_NODES:-1}" -gt 1 ]; then
  # Our own Ray cluster on port 6379. SkyPilot runs its own Ray on 6380; leave it alone.
  if [ "${SKYPILOT_NODE_RANK}" = "0" ]; then
    "$RAY" start --head --port=__RAY_PORT__ --disable-usage-stats --dashboard-host=127.0.0.1 \\
      --num-gpus="${SKYPILOT_NUM_GPUS_PER_NODE:-0}"
    for i in $(seq 1 180); do
      alive=$("$PY" -c 'import ray; ray.init(address="127.0.0.1:__RAY_PORT__", log_to_driver=False); print(sum(1 for n in ray.nodes() if n["Alive"]))' 2>/dev/null || echo 0)
      if [ "$alive" -ge "$SKYPILOT_NUM_NODES" ]; then break; fi
      echo "waiting for workers to join Ray ($alive/$SKYPILOT_NUM_NODES)"; sleep 5
    done
    RAY_ARGS="--ray-address 127.0.0.1:__RAY_PORT__"
  else
    for i in $(seq 1 180); do
      if "$RAY" health-check --address="$HEAD_IP:__RAY_PORT__" >/dev/null 2>&1; then break; fi
      sleep 5
    done
    "$RAY" start --address="$HEAD_IP:__RAY_PORT__" --disable-usage-stats \\
      --num-gpus="${SKYPILOT_NUM_GPUS_PER_NODE:-0}" --block
    exit 0
  fi
fi
exec "$SP" -v serve "$SERVEPILOT_MODEL" --host 0.0.0.0 --port $SERVEPILOT_PORT $RAY_ARGS $SERVEPILOT_ARGS
""".replace("__RAY_PORT__", str(RAY_PORT))


def render_task(req: LaunchRequest) -> dict[str, Any]:
    """The SkyPilot task as a plain dict (see docs.skypilot.co YAML spec)."""
    resources: dict[str, Any] = {
        "infra": req.infra,
        "ports": req.port,
        "use_spot": req.spot,
        "disk_size": req.disk_size_gb,
    }
    if req.instance:
        resources["instance_type"] = req.instance
    if req.accelerators:
        resources["accelerators"] = req.accelerators
    if req.autostop_minutes is not None:
        resources["autostop"] = {"idle_minutes": req.autostop_minutes, "down": True}
    engine_flag = [] if req.engine == "auto" else ["--engine", req.engine]
    task: dict[str, Any] = {
        "name": req.name,
        "num_nodes": req.nodes,
        "resources": resources,
        "envs": {
            "SERVEPILOT_MODEL": req.model,
            "SERVEPILOT_PORT": str(req.port),
            "SERVEPILOT_PACKAGE": req.package,
            "SERVEPILOT_ENGINE": "sglang" if req.engine == "sglang" else "vllm",
            "SERVEPILOT_ENGINE_PACKAGES": ENGINE_PACKAGES[req.engine],
            "SERVEPILOT_ARGS": " ".join(shlex.quote(a) for a in [*engine_flag, *req.serve_args]),
        },
        "setup": SETUP_SCRIPT,
        "run": RUN_SCRIPT,
    }
    if req.hf_token:
        task["secrets"] = {"HF_TOKEN": req.hf_token}
    return task


class _BlockDumper(yaml.SafeDumper):
    """Write multi-line strings (the setup/run scripts) as readable ``|`` blocks."""


def _str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_BlockDumper.add_representer(str, _str_representer)


def render_task_yaml(req: LaunchRequest, *, redact: bool = False) -> str:
    task = render_task(req)
    if redact and "secrets" in task:
        task["secrets"] = dict.fromkeys(task["secrets"], "***")
    return yaml.dump(task, Dumper=_BlockDumper, sort_keys=False, width=1000)


OutputCallback = Callable[[str], None]


@dataclass
class SkyResult:
    returncode: int
    stdout: str
    stderr: str


class SkyClient:
    """Thin wrapper over the ``sky`` CLI. ``executable`` can be swapped in tests."""

    def __init__(
        self, executable: str | None = None, *, on_output: OutputCallback | None = None
    ) -> None:
        self._exe = executable or os.environ.get("SERVEPILOT_SKY_BIN") or shutil.which("sky")
        self._on_output = on_output

    def available(self) -> bool:
        return bool(self._exe) and Path(str(self._exe)).exists()

    def require(self) -> str:
        if not self.available():
            raise CloudError(
                "SkyPilot's `sky` command was not found.",
                hints=[
                    'pip install "servepilot[cloud]"   (installs SkyPilot with AWS, GCP and Azure support)',
                    "then run `sky check` to confirm your cloud credentials",
                ],
            )
        return str(self._exe)

    def command(self, *args: str) -> list[str]:
        return [self.require(), *args]

    def run(self, *args: str, stream: bool = False, timeout: float | None = None) -> SkyResult:
        cmd = self.command(*args)
        log.info("running %s", " ".join(shlex.quote(c) for c in cmd))
        if self._on_output is not None:
            self._on_output("$ " + " ".join(shlex.quote(c) for c in cmd))
        if stream:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
            )
            lines: list[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                lines.append(line)
                if self._on_output is not None:
                    self._on_output(redact_secrets(line))
            proc.wait(timeout=timeout)
            return SkyResult(proc.returncode, "\n".join(lines), "")
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False, env=os.environ.copy()
        )
        return SkyResult(done.returncode, done.stdout, done.stderr)

    # ------------------------------------------------------------------ operations
    def check(self) -> SkyResult:
        return self.run("check")

    def launch(self, task_path: Path, name: str, *, detach: bool = True) -> SkyResult:
        args = ["launch", "-c", name, "-y"]
        if detach:
            args.append("-d")
        args.append(str(task_path))
        result = self.run(*args, stream=True)
        if result.returncode != 0:
            raise CloudError(
                f"`sky launch` failed with exit code {result.returncode}.",
                hints=[
                    "Run `sky check` to verify credentials for this cloud.",
                    "Check quota and availability for the requested instance type/region.",
                    f"Inspect the task file: {task_path}",
                ],
            )
        return result

    def endpoint(self, name: str, port: int) -> str | None:
        result = self.run("status", name, "--endpoint", str(port))
        if result.returncode != 0:
            return None
        for line in reversed(result.stdout.strip().splitlines()):
            line = line.strip()
            if line and ("://" in line or re.match(r"^[\d.]+:\d+$", line)):
                return line if "://" in line else f"http://{line}"
        return None

    def status(self, name: str | None = None) -> list[dict[str, Any]]:
        args = ["status"] + ([name] if name else [])
        result = self.run(*args, "-o", "json")
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout)
                if isinstance(payload, list):
                    return [c for c in payload if isinstance(c, dict)]
                if isinstance(payload, dict) and isinstance(payload.get("clusters"), list):
                    return [c for c in payload["clusters"] if isinstance(c, dict)]
            except json.JSONDecodeError:
                pass
        plain = self.run(*args)
        return [{"raw": plain.stdout}]

    def down(self, name: str) -> SkyResult:
        result = self.run("down", name, "-y", stream=True)
        if result.returncode != 0:
            raise CloudError(f"`sky down {name}` failed with exit code {result.returncode}.")
        return result

    def logs(self, name: str, *, tail: int | None = 100) -> SkyResult:
        args = ["logs", name, "--no-follow"]
        if tail is not None:
            args += ["--tail", str(tail)]
        return self.run(*args)


def wait_for_endpoint(
    client: SkyClient,
    name: str,
    port: int,
    *,
    timeout_seconds: float,
    poll_seconds: float = 15.0,
    probe: Callable[[str], bool] | None = None,
    on_progress: OutputCallback | None = None,
) -> str | None:
    """Poll ``sky status --endpoint`` and then ``/health`` until ServePilot answers, or time out."""
    import httpx

    def default_probe(url: str) -> bool:
        try:
            return httpx.get(f"{url.rstrip('/')}/health", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    probe = probe or default_probe
    deadline = time.monotonic() + timeout_seconds
    url: str | None = None
    while time.monotonic() < deadline:
        if url is None:
            url = client.endpoint(name, port)
        if url is not None and probe(url):
            return url
        if on_progress is not None:
            on_progress(
                "not ready yet (setup, download and tuning can take a while); `sky logs "
                + name
                + "` shows progress"
            )
        time.sleep(poll_seconds)
    return url
