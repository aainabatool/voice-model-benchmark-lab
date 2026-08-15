#!/usr/bin/env python
"""Pareto frontier and ranking analysis for one experiment's STT results.

Usage:
    uv run python scripts/analyze_experiment.py --experiment-id run_...
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from voice_benchmark.analytics.pareto import (
    ParetoPoint,
    compute_pareto_frontier,
    rank_by_weighted_score,
)
from voice_benchmark.storage.db import init_db, session_scope
from voice_benchmark.storage.repositories.experiment_repository import get_experiment
from voice_benchmark.storage.repositories.result_repository import get_results_for_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pareto frontier (accuracy vs. speed) analysis for one experiment."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--accuracy-weight",
        type=float,
        default=0.5,
        help="0.0 = rank by speed only, 1.0 = rank by accuracy only, 0.5 = equal weight.",
    )
    args = parser.parse_args()

    init_db()
    with session_scope() as session:
        experiment = get_experiment(session, args.experiment_id)
        if experiment is None:
            print(f"No experiment found with id '{args.experiment_id}'")
            return
        results = get_results_for_experiment(session, args.experiment_id)

    if not results:
        print(f"Experiment '{args.experiment_id}' has no results yet.")
        return

    by_model: dict[str, list] = defaultdict(list)
    for r in results:
        if not r.failed and r.wer is not None and r.rtf is not None:
            by_model[r.model].append(r)

    if not by_model:
        print("No successful results with WER/RTF to analyze.")
        return

    points = []
    print(f"{'model':<28} {'avg_wer':>10} {'avg_rtf':>10} {'n':>4}")
    for model, model_results in by_model.items():
        avg_wer = sum(r.wer for r in model_results) / len(model_results)
        avg_rtf = sum(r.rtf for r in model_results) / len(model_results)
        print(f"{model:<28} {avg_wer:>10.4f} {avg_rtf:>10.4f} {len(model_results):>4}")
        points.append(ParetoPoint(model, x=avg_wer, y=avg_rtf))

    frontier = compute_pareto_frontier(points)
    print(f"\nPareto-optimal models (no other model beats them on both WER and RTF):")
    for name in frontier:
        print(f"  * {name}")

    ranked = rank_by_weighted_score(points, accuracy_weight=args.accuracy_weight)
    print(f"\nRanked (accuracy_weight={args.accuracy_weight}, lower score = better):")
    for i, (name, score) in enumerate(ranked, start=1):
        marker = " [Pareto-optimal]" if name in frontier else ""
        print(f"  {i}. {name}: {score:.4f}{marker}")


if __name__ == "__main__":
    main()
