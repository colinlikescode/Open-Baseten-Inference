"""Cloud commands: launch on a cloud through SkyPilot, list clusters, tear them down, show shapes."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from typing import Annotated, Any, Protocol, TypeVar

import typer
from rich.table import Table

from servepilot.cli.common import (
    AcceleratorsOpt,
    ExpectedConcurrencyOpt,
    InstanceOpt,
    JSONOpt,
    MaxLatencyOpt,
    MaxTPOTOpt,
    MaxTTFTOpt,
    NodesOpt,
    ObjectiveOpt,
    PlanFlags,
    ProfileOpt,
    build_workspace,
    emit_json,
    get_state,
    hf_token,
)
from servepilot.cli.pipeline import run_plan
from servepilot.cli.render import render_hardware, render_model, render_plan, render_workload
from servepilot.cloud.catalog import PROVIDERS, list_shapes
from servepilot.cloud.skypilot import LaunchRequest, SkyClient, render_task_yaml, wait_for_endpoint
from servepilot.exceptions import ConfigurationError

# Only one engine is installed on the rented machines; "auto" means vLLM there.
LaunchEngineOpt = Annotated[
    str | None,
    typer.Option(
        "--engine",
        help="vllm | sglang. The machines get only this engine (default: vllm).",
        rich_help_panel="Basics",
    ),
]


def _installed_engine(engine: str | None) -> str:
    """The engine `launch` installs remotely; the local pre-launch plan must use the same one."""
    return "vllm" if engine in (None, "auto") else engine


def _serve_args(
    objective: str | None,
    profile: str | None,
    expected_concurrency: int | None,
    ttft: float | None,
    latency: float | None,
    tpot: float | None,
) -> list[str]:
    args: list[str] = []
    if objective:
        args += ["--objective", objective]
    if profile:
        args += ["--profile", profile]
    if expected_concurrency:
        args += ["--expected-concurrency", str(expected_concurrency)]
    if ttft:
        args += ["--max-p95-ttft", str(ttft)]
    if latency:
        args += ["--max-p95-latency", str(latency)]
    if tpot:
        args += ["--max-p95-tpot", str(tpot)]
    return args


F = TypeVar("F", bound=Callable[..., Any])


class ErrorHandler(Protocol):
    def __call__(self, func: F) -> F: ...


def register(app: typer.Typer, inspect_app: typer.Typer, handle_errors: ErrorHandler) -> None:
    @app.command()
    @handle_errors
    def launch(
        ctx: typer.Context,
        model: Annotated[str, typer.Argument(help="Hugging Face model id.", show_default=False)],
        cloud: Annotated[
            str,
            typer.Option(
                "--cloud",
                help="aws | gcp | azure (your own account, provisioned through SkyPilot).",
                rich_help_panel="Basics",
            ),
        ],
        instance: InstanceOpt = None,
        accelerators: AcceleratorsOpt = None,
        nodes: NodesOpt = 1,
        name: Annotated[
            str | None,
            typer.Option(
                "--name",
                help="Cluster name (default: servepilot-<model>).",
                rich_help_panel="Basics",
            ),
        ] = None,
        region: Annotated[
            str | None,
            typer.Option(
                "--region", help="Cloud region, like us-east-1.", rich_help_panel="Basics"
            ),
        ] = None,
        engine: LaunchEngineOpt = None,
        objective: ObjectiveOpt = None,
        profile: ProfileOpt = None,
        expected_concurrency: ExpectedConcurrencyOpt = None,
        max_p95_ttft: MaxTTFTOpt = None,
        max_p95_latency: MaxLatencyOpt = None,
        max_p95_tpot: MaxTPOTOpt = None,
        spot: Annotated[
            bool,
            typer.Option(
                "--spot", help="Use spot/preemptible machines.", rich_help_panel="Advanced"
            ),
        ] = False,
        autostop: Annotated[
            int | None,
            typer.Option(
                "--autostop",
                help="Tear the cluster down after this many idle minutes once serving stops.",
                rich_help_panel="Advanced",
            ),
        ] = None,
        disk_size: Annotated[
            int,
            typer.Option(
                "--disk-size",
                help="Disk per node in GB (models are big).",
                rich_help_panel="Advanced",
            ),
        ] = 512,
        package: Annotated[
            str | None,
            typer.Option(
                "--package",
                help="pip spec for ServePilot on the nodes (default: this version from PyPI).",
                rich_help_panel="Advanced",
            ),
        ] = None,
        no_wait: Annotated[
            bool,
            typer.Option(
                "--no-wait",
                help="Return right after `sky launch`; do not wait for the endpoint.",
                rich_help_panel="Advanced",
            ),
        ] = False,
        timeout: Annotated[
            float,
            typer.Option(
                "--timeout",
                help="Seconds to wait for the endpoint to become healthy.",
                rich_help_panel="Advanced",
            ),
        ] = 5400.0,
        dry_run: Annotated[
            bool,
            typer.Option(
                "--dry-run",
                help="Show the plan and the SkyPilot task; launch nothing.",
                rich_help_panel="Basics",
            ),
        ] = False,
        json_output: JSONOpt = False,
    ) -> None:
        """Rent machines through SkyPilot and run `servepilot serve` on them."""
        state = get_state(ctx)
        console = state.err_console if json_output else state.console
        serve_args = _serve_args(
            objective, profile, expected_concurrency, max_p95_ttft, max_p95_latency, max_p95_tpot
        )
        request_kwargs: dict[str, Any] = {
            "model": model,
            "cloud": cloud,
            "instance": instance,
            "accelerators": accelerators,
            "nodes": nodes,
            "name": name,
            "region": region,
            "engine": engine or "auto",
            "spot": spot,
            "autostop_minutes": autostop,
            "disk_size_gb": disk_size,
            "serve_args": serve_args,
            "hf_token": hf_token(),
        }
        if package:
            request_kwargs["package"] = package
        request = LaunchRequest(**request_kwargs)

        # Static plan on the catalog shape, so you see what will be tried before paying.
        planning_note: str | None = None
        try:
            flags = PlanFlags(
                model=model,
                engine=_installed_engine(engine),
                objective=objective,
                profile=profile,
                expected_concurrency=expected_concurrency,
                max_p95_ttft=max_p95_ttft,
                max_p95_latency=max_p95_latency,
                max_p95_tpot=max_p95_tpot,
                cloud=cloud,
                instance=instance,
                accelerators=accelerators,
                nodes=nodes,
            )
            ws = build_workspace(flags, state)
            planning, explanation = run_plan(ws)
            render_hardware(console, ws.hardware)
            console.print()
            render_model(console, ws.model)
            console.print()
            render_workload(console, ws.workload)
            render_plan(console, planning, explanation)
        except ConfigurationError as exc:
            planning_note = f"could not plan ahead of time: {exc.message}"
            console.print(f"[yellow]![/] {planning_note}")
            planning = None

        task_yaml = render_task_yaml(request, redact=True)
        task_dir = state.settings.cache_dir / "skypilot"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_path = task_dir / f"{request.name}.yaml"
        task_path.write_text(render_task_yaml(request), encoding="utf-8")
        task_path.chmod(0o600)

        client = SkyClient(
            on_output=lambda line: console.print(f"[dim]{line}[/]", markup=False, highlight=False)
        )
        launch_cmd = " ".join(
            shlex.quote(c)
            for c in ["sky", "launch", "-c", str(request.name), "-y", "-d", str(task_path)]
        )
        console.print()
        console.print(f"[bold]SkyPilot task[/] written to {task_path}")
        console.print(task_yaml, markup=False, highlight=False)
        console.print(f"[bold]Command:[/] {launch_cmd}")
        if dry_run:
            if json_output:
                emit_json(
                    {
                        "name": request.name,
                        "task_path": str(task_path),
                        "task": task_yaml,
                        "command": launch_cmd,
                        "planning": planning.model_dump(mode="json") if planning else None,
                        "note": planning_note,
                    }
                )
            return

        client.require()
        console.print(
            f"\nLaunching {nodes} × {instance or accelerators} on {request.infra} as [bold]{request.name}[/] ..."
        )
        client.launch(task_path, str(request.name), detach=True)
        endpoint: str | None = None
        if not no_wait:
            endpoint = wait_for_endpoint(
                client,
                str(request.name),
                request.port,
                timeout_seconds=timeout,
                on_progress=lambda msg: console.print(f"[dim]{msg}[/]"),
            )
        result: dict[str, Any] = {
            "name": request.name,
            "cloud": request.infra,
            "nodes": nodes,
            "task_path": str(task_path),
            "endpoint": endpoint,
            "logs": f"sky logs {request.name}",
            "down": f"servepilot down {request.name}",
        }
        if json_output:
            emit_json(result)
            return
        console.print()
        if endpoint:
            console.print(f"[green]Serving.[/] OpenAI-compatible API: {endpoint.rstrip('/')}/v1")
        else:
            console.print(
                "[yellow]Not healthy yet.[/] Setup, model download and tuning can take a while."
            )
            console.print(f"  follow along: sky logs {request.name}")
            console.print(f"  endpoint:     sky status {request.name} --endpoint {request.port}")
        console.print(f"  tear down:    servepilot down {request.name}")

    @app.command()
    @handle_errors
    def down(
        ctx: typer.Context,
        name: Annotated[
            str,
            typer.Argument(help="Cluster name from `servepilot launch` or `servepilot clusters`."),
        ],
        json_output: JSONOpt = False,
    ) -> None:
        """Tear down a cluster started with `servepilot launch` (runs `sky down`)."""
        state = get_state(ctx)
        console = state.err_console if json_output else state.console
        client = SkyClient(
            on_output=lambda line: console.print(f"[dim]{line}[/]", markup=False, highlight=False)
        )
        client.down(name)
        if json_output:
            emit_json({"name": name, "down": True})
        else:
            console.print(f"[green]{name} is gone.[/]")

    @app.command()
    @handle_errors
    def clusters(ctx: typer.Context, json_output: JSONOpt = False) -> None:
        """List SkyPilot clusters (runs `sky status`)."""
        state = get_state(ctx)
        client = SkyClient()
        entries = client.status()
        if json_output:
            emit_json(entries)
            return
        if entries and "raw" in entries[0]:
            state.console.print(entries[0]["raw"], markup=False, highlight=False)
            return
        table = Table(title="SkyPilot clusters", expand=False)
        for col in ("Name", "Status", "Infra", "Resources", "Launched"):
            table.add_column(col)
        for c in entries:
            table.add_row(
                str(c.get("name", "")),
                str(c.get("status", "")),
                str(c.get("infra") or c.get("cloud") or ""),
                str(c.get("resources_str") or c.get("resources") or ""),
                str(c.get("launched_at") or ""),
            )
        state.console.print(table)
        state.console.print(
            "Endpoint of one cluster: sky status NAME --endpoint 8000   ·   tear down: servepilot down NAME"
        )

    @inspect_app.command("cloud")
    @handle_errors
    def inspect_cloud(
        ctx: typer.Context,
        provider: Annotated[
            str | None, typer.Argument(help="aws | gcp | azure (default: all)")
        ] = None,
        json_output: JSONOpt = False,
    ) -> None:
        """Instance shapes ServePilot can plan for (GPU model, count, memory)."""
        state = get_state(ctx)
        shapes = list_shapes(provider)
        if json_output:
            emit_json(
                [
                    {
                        "provider": s.provider,
                        "instance": s.name,
                        "gpu": s.gpu_model.name,
                        "gpu_count": s.gpu_count,
                        "gpu_memory_bytes": s.gpu_model.memory_bytes,
                        "nvlink": s.nvlink,
                    }
                    for s in shapes
                ]
            )
            return
        table = Table(
            title="Cloud instance shapes" + (f" ({provider})" if provider else ""), expand=False
        )
        for col in ("Cloud", "Instance", "GPUs", "GPU", "Memory/GPU", "NVLink"):
            table.add_column(col)
        for s in shapes:
            table.add_row(
                s.provider,
                s.name,
                str(s.gpu_count),
                s.gpu_model.name,
                f"{s.gpu_model.memory_mib / 1024:.0f} GiB",
                "yes" if s.nvlink else "no",
            )
        state.console.print(table)
        state.console.print(
            f"Clouds with shapes: {', '.join(PROVIDERS)}. Any GPU works with --accelerators H100:8 style input."
        )
