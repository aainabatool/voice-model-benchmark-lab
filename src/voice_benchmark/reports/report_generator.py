"""Markdown report generation for a benchmark experiment.

Turns an Experiment + its results into a single shareable Markdown
document -- the same aggregations used by the dashboard (per-model
summary, Pareto frontier, robustness-by-condition breakdown), just
rendered as plain-text tables so it's readable in any editor, on
GitHub, or printed to a terminal, with no extra dependencies.
"""
from __future__ import annotations

from collections import defaultdict

from voice_benchmark.analytics.pareto import ParetoPoint, compute_pareto_frontier
from voice_benchmark.core.models import Experiment, STTResult

_CONDITION_ORDER = [
    "noise_neg5db",
    "noise_0db",
    "noise_5db",
    "noise_10db",
    "noise_20db",
    "clean",
]


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _per_model_summary(results: list[STTResult]) -> dict[str, dict[str, float | int]]:
    by_model: dict[str, list[STTResult]] = defaultdict(list)
    for r in results:
        by_model[r.model].append(r)

    summary: dict[str, dict[str, float | int]] = {}
    for model, model_results in by_model.items():
        ok = [r for r in model_results if not r.failed and r.wer is not None]
        summary[model] = {
            "avg_wer": sum(r.wer for r in ok) / len(ok) if ok else None,
            "avg_cer": sum(r.cer for r in ok) / len(ok) if ok else None,
            "avg_rtf": sum(r.rtf for r in ok if r.rtf is not None) / len(ok) if ok else None,
            "avg_latency_ms": (
                sum(r.latency_ms for r in ok if r.latency_ms is not None) / len(ok) if ok else None
            ),
            "failures": sum(1 for r in model_results if r.failed),
            "total": len(model_results),
        }
    return summary


def generate_markdown_report(experiment: Experiment, results: list[STTResult]) -> str:
    lines: list[str] = []
    lines.append(f"# Benchmark Report: {experiment.benchmark_name}")
    lines.append("")
    lines.append(f"- **Experiment ID:** `{experiment.experiment_id}`")
    lines.append(f"- **Status:** {experiment.status.value}")
    lines.append(f"- **Started:** {experiment.start_time}")
    lines.append(f"- **Ended:** {experiment.end_time or '-'}")
    lines.append(f"- **Dataset version:** {experiment.dataset_version}")
    lines.append(
        f"- **Conditions tested:** {', '.join(c.value for c in experiment.conditions) or '-'}"
    )
    lines.append("")

    if experiment.hardware:
        hw = experiment.hardware
        gpu = f"{hw.gpu} ({hw.vram_gb}GB VRAM)" if hw.gpu else "none (CPU only)"
        lines.append("## Hardware")
        lines.append("")
        lines.append(f"- OS: {hw.os}")
        lines.append(f"- Python: {hw.python_version}")
        lines.append(f"- CPU: {hw.cpu or '-'}")
        lines.append(f"- RAM: {hw.ram_gb}GB")
        lines.append(f"- GPU: {gpu}")
        lines.append("")

    if not results:
        lines.append("_No results recorded for this experiment._")
        return "\n".join(lines)

    summary = _per_model_summary(results)

    lines.append("## Per-model summary")
    lines.append("")
    lines.append("| Model | Avg WER | Avg CER | Avg RTF | Avg Latency (ms) | Failures | N |")
    lines.append("|---|---|---|---|---|---|---|")
    for model, s in summary.items():
        lines.append(
            f"| {model} | {_fmt(s['avg_wer'])} | {_fmt(s['avg_cer'])} | {_fmt(s['avg_rtf'])} | "
            f"{_fmt(s['avg_latency_ms'], 1)} | {s['failures']} | {s['total']} |"
        )
    lines.append("")

    # Pareto frontier -- only meaningful with 2+ models that have both metrics.
    points = [
        ParetoPoint(model, x=s["avg_wer"], y=s["avg_rtf"])
        for model, s in summary.items()
        if s["avg_wer"] is not None and s["avg_rtf"] is not None
    ]
    if len(points) > 1:
        frontier = compute_pareto_frontier(points)
        lines.append("## Pareto-optimal models (accuracy vs. speed)")
        lines.append("")
        lines.append(
            "No other model beats these on both average WER and average RTF "
            "at the same time:"
        )
        lines.append("")
        for name in frontier:
            lines.append(f"- **{name}**")
        lines.append("")

    # Robustness breakdown -- only meaningful with 2+ distinct conditions.
    conditions_present = {r.condition.value for r in results}
    if len(conditions_present) > 1:
        by_condition_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in results:
            if not r.failed and r.wer is not None:
                by_condition_model[r.condition.value][r.model].append(r.wer)

        model_names = sorted(summary.keys())
        ordered_conditions = [c for c in _CONDITION_ORDER if c in conditions_present]

        lines.append("## Robustness breakdown (WER by noise condition)")
        lines.append("")
        lines.append("| Condition | " + " | ".join(model_names) + " |")
        lines.append("|---|" + "---|" * len(model_names))
        for condition in ordered_conditions:
            row = [condition]
            for model in model_names:
                wers = by_condition_model.get(condition, {}).get(model, [])
                row.append(_fmt(sum(wers) / len(wers)) if wers else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## All results")
    lines.append("")
    lines.append("| Model | Test Case | Condition | WER | CER | RTF | Latency (ms) | Failed |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.model} | {r.test_case_id} | {r.condition.value} | {_fmt(r.wer)} | "
            f"{_fmt(r.cer)} | {_fmt(r.rtf)} | {_fmt(r.latency_ms, 1)} | {r.failed} |"
        )
    lines.append("")

    return "\n".join(lines)
