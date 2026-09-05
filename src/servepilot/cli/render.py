"""Rich rendering helpers for CLI output."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from servepilot.planner.memory import format_bytes
from servepilot.schemas.benchmark import BenchmarkResult, ParetoPoint
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidateEvaluation, PlanningResult, SelectedPlan
from servepilot.schemas.workload import WorkloadProfile


def _fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f} ms"


def _fmt_tps(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


def render_hardware(console: Console, hw: HardwareSnapshot) -> None:
    title = "Hardware"
    if hw.is_cluster:
        title += f" ({len(hw.nodes)} nodes via Ray)"
    table = Table(title=title, show_lines=False, expand=False)
    table.add_column("GPU", justify="right")
    if hw.is_cluster:
        table.add_column("Node")
    table.add_column("Name")
    table.add_column("Memory", justify="right")
    table.add_column("Free", justify="right")
    table.add_column("CC", justify="center")
    table.add_column("NUMA", justify="center")
    for g in hw.gpus:
        row = [str(g.index)]
        if hw.is_cluster:
            row.append(f"{g.node_ip}:{g.device_index_on_node}")
        row += [
            g.name,
            format_bytes(g.total_memory_bytes),
            format_bytes(g.free_memory_bytes),
            g.compute_capability or "-",
            str(g.numa_node) if g.numa_node is not None else "-",
        ]
        table.add_row(*row)
    console.print(table)
    details = [
        f"driver {hw.driver_version or 'unknown'}",
        f"CUDA {hw.cuda_version or 'unknown'}",
        f"topology: {hw.topology.summary()}",
    ]
    if not hw.topology.available and hw.topology.error:
        details.append(f"topology error: {hw.topology.error}")
    console.print("  " + " · ".join(details))
    if hw.gpu_count > 1 and hw.topology.available and hw.topology.edges:
        pairs = ", ".join(
            f"{e.gpu_a}-{e.gpu_b}: {e.relationship}{'' if not e.nvlink_link_count else f' ({e.nvlink_link_count} links)'}"
            for e in hw.topology.edges[:12]
        )
        console.print(f"  pairs: {pairs}{' …' if len(hw.topology.edges) > 12 else ''}")


def render_model(console: Console, model: ModelProfile) -> None:
    table = Table(title=f"Model: {model.model_id}", show_header=False, expand=False)
    table.add_column("field", style="dim")
    table.add_column("value")
    rows = [
        ("architecture", ", ".join(model.architecture_names) or "unknown"),
        ("type", f"{model.architecture_summary} ({model.model_type or 'unknown model_type'})"),
        ("attention", model.attention_kind.value),
        ("layers / hidden", f"{model.num_hidden_layers} / {model.hidden_size}"),
        (
            "heads / kv heads / head_dim",
            f"{model.num_attention_heads} / {model.num_key_value_heads} / {model.head_dim}",
        ),
        ("vocab", str(model.vocab_size)),
        ("max positions", str(model.max_position_embeddings)),
        ("dtype", model.configured_dtype or "unknown"),
        ("quantization", model.quantization_method or "none"),
        (
            "weights",
            f"{format_bytes(model.weight_bytes)} ({model.weight_size_source}{'' if model.weight_bytes_is_exact else ', approximate'})",
        ),
        (
            "parameters (est.)",
            f"{model.estimated_parameter_count / 1e9:.1f}B"
            if model.estimated_parameter_count
            else "unknown",
        ),
        ("revision", model.revision or "-"),
        ("tokenizer", "available" if model.tokenizer_available else "not found"),
        ("trust_remote_code", "required" if model.trust_remote_code_required else "not required"),
    ]
    if model.is_moe:
        rows.append(
            ("experts", f"{model.num_experts} total, {model.num_experts_per_token} per token")
        )
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)
    for w in model.warnings:
        console.print(f"  [yellow]![/] {w}")


def render_workload(console: Console, workload: WorkloadProfile) -> None:
    parts = [
        f"profile {workload.name}",
        f"objective {workload.objective.value}",
        f"input p50/p95 {workload.input_tokens_p50}/{workload.input_tokens_p95}",
        f"output p50/p95 {workload.output_tokens_p50}/{workload.output_tokens_p95}",
        f"context {workload.max_context_tokens}",
        "streaming" if workload.streaming else "non-streaming",
    ]
    if workload.expected_concurrency:
        parts.append(f"expected concurrency {workload.expected_concurrency}")
    if workload.has_slo and workload.latency_constraints is not None:
        slo = workload.latency_constraints
        slo_parts = []
        if slo.max_p95_ttft_ms:
            slo_parts.append(f"p95 TTFT ≤ {slo.max_p95_ttft_ms:.0f} ms")
        if slo.max_p95_latency_ms:
            slo_parts.append(f"p95 latency ≤ {slo.max_p95_latency_ms:.0f} ms")
        if slo.max_p95_tpot_ms:
            slo_parts.append(f"p95 TPOT ≤ {slo.max_p95_tpot_ms:.1f} ms")
        parts.append("SLO " + ", ".join(slo_parts))
    console.print("Workload: " + " · ".join(parts))


def render_plan(console: Console, result: PlanningResult, explanation: Sequence[str]) -> None:
    console.print()
    if result.estimated_minimum_tp is not None:
        console.print(f"Estimated minimum TP: [bold]{result.estimated_minimum_tp}[/]")
    table = Table(title="Candidates", expand=False)
    table.add_column("#", justify="right")
    table.add_column("Topology")
    table.add_column("GPU groups")
    table.add_column("Weights/GPU", justify="right")
    table.add_column("KV/GPU", justify="right")
    table.add_column("~Conc./replica", justify="right")
    table.add_column("Mem frac", justify="right")
    table.add_column("Viability")
    for i, p in enumerate(result.candidates, start=1):
        est = p.estimated_memory
        groups = ", ".join(str(g) for g in p.gpu_groups[:4]) + (
            " …" if len(p.gpu_groups) > 4 else ""
        )
        table.add_row(
            str(i),
            p.label(),
            groups,
            format_bytes(est.weights_bytes) if est else "-",
            format_bytes(est.kv_cache_bytes_available)
            if est and est.kv_cache_bytes_available is not None
            else "?",
            str(est.estimated_max_concurrency_p50)
            if est and est.estimated_max_concurrency_p50 is not None
            else "?",
            f"{p.memory_fraction:.2f}" if p.memory_fraction is not None else "-",
            p.viability.value.replace("_", " "),
        )
    console.print(table)
    for i, p in enumerate(result.candidates, start=1):
        if p.rationale:
            console.print(f"  [bold]{i}. {p.label()}[/]")
            for line in p.rationale[:4]:
                console.print(f"     - {line}")
    if result.excluded:
        console.print()
        console.print("[bold]Excluded[/]")
        for e in result.excluded:
            console.print(f"  - {e.description}: {e.reason}")
    for w in result.warnings:
        console.print(f"  [yellow]![/] {w}")
    for n in result.notes:
        console.print(f"  [dim]{n}[/]")
    if explanation:
        console.print()
        console.print(Panel("\n".join(explanation), title="Why", expand=False))


def render_results_table(
    console: Console, evaluations: Sequence[CandidateEvaluation], title: str = "Benchmarks"
) -> None:
    table = Table(title=title, expand=False)
    table.add_column("Candidate")
    table.add_column("Stage")
    table.add_column("Conc.", justify="right")
    table.add_column("Output tok/s", justify="right")
    table.add_column("Req/s", justify="right")
    table.add_column("p95 TTFT", justify="right")
    table.add_column("p95 TPOT", justify="right")
    table.add_column("p95 latency", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Status")
    for e in evaluations:
        if e.status == "failed":
            table.add_row(
                e.plan.label(),
                e.stage,
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                f"[red]{e.failure.type.value if e.failure else 'failed'}[/]",
            )
            continue
        for r in e.results:
            table.add_row(
                e.plan.label(),
                f"{e.stage}{'/' + r.spec.label if r.spec.label else ''}",
                str(r.spec.concurrency),
                _fmt_tps(r.output_tokens_per_second),
                f"{r.request_throughput:.2f}",
                _fmt_ms(r.ttft_p95_ms),
                f"{r.tpot_p95_ms:.1f} ms" if r.tpot_p95_ms is not None else "-",
                _fmt_ms(r.latency_p95_ms),
                f"{r.error_rate:.1%}",
                "ok",
            )
    console.print(table)


def render_pareto(console: Console, points: Sequence[ParetoPoint]) -> None:
    table = Table(title="Pareto frontier (throughput vs p95 latency)", expand=False)
    table.add_column("Candidate")
    table.add_column("Concurrency", justify="right")
    table.add_column("Output tok/s", justify="right")
    table.add_column("p95 latency", justify="right")
    table.add_column("Front")
    for p in points:
        table.add_row(
            p.candidate_id,
            str(p.concurrency),
            _fmt_tps(p.output_tokens_per_second),
            _fmt_ms(p.latency_p95_ms),
            "[green]✓[/]" if p.on_front else "",
        )
    console.print(table)


def render_selected(
    console: Console, selected: SelectedPlan, *, commands: Sequence[str] = ()
) -> None:
    p = selected.plan
    lines = [
        f"[bold]{p.label()}[/]",
        f"engine: {p.engine.value}"
        + (f" {selected.engine_version}" if selected.engine_version else ""),
        f"GPU groups: {p.gpu_groups}",
        f"context length: {p.context_length}",
        f"memory fraction: {p.memory_fraction}",
        f"max concurrency: {p.max_concurrency} (per replica max sequences: {p.max_num_seqs})",
        f"source: {selected.source}"
        + ("" if selected.benchmarked else "  [yellow](UNBENCHMARKED heuristic)[/]"),
    ]
    if selected.final_result is not None:
        r = selected.final_result
        lines.append(
            f"measured: {r.output_tokens_per_second:,.0f} output tok/s · p95 TTFT {_fmt_ms(r.ttft_p95_ms)} · "
            f"p95 TPOT {f'{r.tpot_p95_ms:.1f} ms' if r.tpot_p95_ms is not None else '-'} · p95 latency {_fmt_ms(r.latency_p95_ms)} "
            f"· errors {r.error_rate:.1%} (concurrency {r.spec.concurrency}, {r.total_requests} requests)"
        )
    if selected.slo_satisfied is False:
        lines.append("[red]SLO not satisfied by any tested configuration[/]")
    console.print(Panel("\n".join(lines), title="Selected plan", expand=False))
    if selected.rationale:
        console.print("[bold]Why ServePilot selected this configuration[/]")
        for i, line in enumerate(selected.rationale, start=1):
            console.print(f"  {i}. {line}")
    if commands:
        console.print()
        console.print("[bold]Equivalent backend configuration[/]")
        for c in commands:
            console.print(f"  {c}", highlight=False, markup=False)


def summarize_result(r: BenchmarkResult) -> str:
    return (
        f"{r.output_tokens_per_second:,.0f} tok/s · p95 TTFT {_fmt_ms(r.ttft_p95_ms)} · p95 latency {_fmt_ms(r.latency_p95_ms)} · "
        f"errors {r.error_rate:.1%}"
    )
