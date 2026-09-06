"""Generate human-readable explanations from planning and tuning data.

Nothing here invents facts: every sentence is derived from a memory estimate, a benchmark result
or a recorded failure.
"""

from __future__ import annotations

from collections.abc import Sequence

from servepilot.planner.memory import format_bytes
from servepilot.schemas.benchmark import BenchmarkResult
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidateEvaluation, CandidatePlan, PlanningResult
from servepilot.schemas.workload import Objective


def explain_plan(
    result: PlanningResult, model: ModelProfile, hardware: HardwareSnapshot
) -> list[str]:
    """Static explanation for ``servepilot plan``."""
    lines: list[str] = []
    gpu = hardware.gpus[0] if hardware.gpus else None
    if result.estimated_minimum_tp is not None and gpu is not None:
        tp = result.estimated_minimum_tp
        if tp == 1:
            lines.append(
                f"Why TP=1 is viable: model weights ({format_bytes(model.weight_bytes)}, {model.weight_size_source}) "
                f"fit within one {format_bytes(gpu.total_memory_bytes)} device with the requested context and headroom."
            )
        else:
            lines.append(
                f"Why TP≥{tp}: model weights ({format_bytes(model.weight_bytes)}, {model.weight_size_source}) cannot safely "
                f"fit on one {format_bytes(gpu.total_memory_bytes)} device at the requested precision and context; "
                f"TP={tp} is the smallest viable tensor-parallel size."
            )
    viable = result.viable
    if len(viable) > 1:
        lines.append(
            f"{len(result.selected_gpu_ids)} GPUs are selected. Viable layouts for benchmarking:"
        )
        for i, p in enumerate(viable):
            lines.append(f"  {chr(ord('A') + i) if i < 26 else i}. {p.label()}")
        lines.append("The final topology will be selected by benchmarking, not by these estimates.")
    elif len(viable) == 1:
        lines.append(
            f"Only one structural topology is viable ({viable[0].label()}); tuning will focus on concurrency and memory settings."
        )
    return lines


def _pct(new: float, old: float) -> str:
    if old <= 0:
        return "n/a"
    return f"{(new - old) / old:+.1%}"


def _key_of(by_key: dict[str, list[BenchmarkResult]], result: BenchmarkResult) -> str:
    for key, results in by_key.items():
        if any(r is result for r in results):
            return key
    return ""


def _best_point(results: Sequence[BenchmarkResult], objective: Objective) -> BenchmarkResult | None:
    """The best observed operating point of one structural candidate (used for fair comparisons)."""
    valid = [r for r in results if r.successful_requests > 0 and r.spec.label != "final"]
    if not valid:
        return None
    if objective == Objective.LATENCY:
        return min(valid, key=lambda r: r.latency_p95_ms)
    return max(valid, key=lambda r: r.output_tokens_per_second)


def build_rationale(
    *,
    winner: CandidatePlan,
    evaluations: Sequence[CandidateEvaluation],
    objective: Objective,
    sweep_history: Sequence[BenchmarkResult],
    chosen_concurrency_reason: str | None,
    final_result: BenchmarkResult | None,
    slo_satisfied: bool | None,
    memory_tuning_note: str | None = None,
    smaller_tp_excluded: bool = False,
) -> list[str]:
    """Explain a tuning outcome from observed data."""
    lines: list[str] = []
    est = winner.estimated_memory
    if est is not None and est.weights_bytes is not None:
        if winner.tensor_parallel_size == 1:
            lines.append(
                f"TP=1 is memory-safe for this model ({format_bytes(est.weights_bytes)} of weights per GPU)."
            )
        elif smaller_tp_excluded:
            lines.append(
                f"TP={winner.tensor_parallel_size} keeps weights at ~{format_bytes(est.weights_bytes)} per GPU; smaller TP sizes did not fit statically."
            )
        else:
            lines.append(
                f"TP={winner.tensor_parallel_size} keeps weights at ~{format_bytes(est.weights_bytes)} per GPU."
            )
    if winner.replica_count > 1:
        lines.append(
            f"TP={winner.tensor_parallel_size} permits {winner.replica_count} independent replicas across {winner.gpu_count} GPUs."
        )

    # Compare each structural candidate at its own best observed operating point.
    by_key: dict[str, list[BenchmarkResult]] = {}
    labels: dict[str, str] = {}
    engines_by_key: dict[str, str] = {}
    for e in evaluations:
        if e.stage in ("structural", "concurrency") and e.results:
            key = e.plan.structural_key()
            by_key.setdefault(key, []).extend(e.results)
            labels[key] = e.plan.label()
            engines_by_key[key] = e.plan.engine.value
    winner_key = winner.structural_key()
    winner_best = _best_point(by_key.get(winner_key, []), objective)
    if winner_best is not None and len(by_key) > 1:
        for key, results in by_key.items():
            if key == winner_key:
                continue
            other = _best_point(results, objective)
            if other is None:
                continue
            if objective == Objective.LATENCY:
                lines.append(
                    f"{labels[key]} reached p95 latency {other.latency_p95_ms:,.0f} ms at best (concurrency {other.spec.concurrency}) vs "
                    f"{winner_best.latency_p95_ms:,.0f} ms for the winner ({_pct(other.latency_p95_ms, winner_best.latency_p95_ms)})."
                )
            else:
                lines.append(
                    f"{labels[key]} peaked at {other.output_tokens_per_second:,.0f} output tok/s (concurrency {other.spec.concurrency}) vs "
                    f"{winner_best.output_tokens_per_second:,.0f} for the winner ({_pct(other.output_tokens_per_second, winner_best.output_tokens_per_second)})."
                )
        other_engines = {
            engines_by_key[k] for k in by_key if engines_by_key[k] != winner.engine.value
        }
        if other_engines and objective != Objective.LATENCY:
            best_other = max(
                (
                    p
                    for k, rs in by_key.items()
                    if engines_by_key[k] != winner.engine.value
                    and (p := _best_point(rs, objective)) is not None
                ),
                key=lambda r: r.output_tokens_per_second,
                default=None,
            )
            if best_other is not None:
                lines.append(
                    f"{winner.engine.value} outperformed the best tested {engines_by_key[_key_of(by_key, best_other)]} candidate by "
                    f"{_pct(winner_best.output_tokens_per_second, best_other.output_tokens_per_second)} output tok/s."
                )

    failed = [e for e in evaluations if e.status == "failed" and e.failure is not None]
    for e in failed[:4]:
        assert e.failure is not None
        lines.append(
            f"{e.plan.label()} failed to launch ({e.failure.type.value}): {e.failure.message.splitlines()[0][:140]}"
        )

    if len(sweep_history) > 1:
        pts = sorted(sweep_history, key=lambda r: r.spec.concurrency)
        best = max(pts, key=lambda r: r.output_tokens_per_second)
        lines.append(
            f"Concurrency was swept from {pts[0].spec.concurrency} to {pts[-1].spec.concurrency}; throughput peaked at "
            f"{best.output_tokens_per_second:,.0f} tok/s (concurrency {best.spec.concurrency})."
        )
    if chosen_concurrency_reason:
        lines.append(chosen_concurrency_reason[0].upper() + chosen_concurrency_reason[1:] + ".")
    if memory_tuning_note:
        lines.append(memory_tuning_note)
    if final_result is not None:
        ttft = (
            f", p95 TTFT {final_result.ttft_p95_ms:,.0f} ms"
            if final_result.ttft_p95_ms is not None
            else ""
        )
        lines.append(
            f"Final confirmation run ({final_result.total_requests} requests at concurrency {final_result.spec.concurrency}): "
            f"{final_result.output_tokens_per_second:,.0f} output tok/s, p95 latency {final_result.latency_p95_ms:,.0f} ms{ttft}, "
            f"error rate {final_result.error_rate:.1%}."
        )
    if slo_satisfied is False:
        lines.append(
            "No tested configuration satisfied the requested SLO; this is the best observed configuration."
        )
    return lines
