#!/usr/bin/env python
"""List experiments and their results from the database.

Usage:
    uv run python scripts/list_experiments.py
    uv run python scripts/list_experiments.py --experiment-id run_20260813T103500Z
"""
from __future__ import annotations

import argparse

from voice_benchmark.storage.db import init_db, session_scope
from voice_benchmark.storage.repositories.experiment_repository import (
    get_experiment,
    list_experiments,
)
from voice_benchmark.storage.repositories.result_repository import get_results_for_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="List persisted experiments and results.")
    parser.add_argument("--experiment-id", default=None, help="Show results for one experiment.")
    args = parser.parse_args()

    init_db()

    with session_scope() as session:
        if args.experiment_id:
            experiment = get_experiment(session, args.experiment_id)
            if experiment is None:
                print(f"No experiment found with id '{args.experiment_id}'")
                return
            results = get_results_for_experiment(session, args.experiment_id)
            print(f"{experiment.experiment_id}  [{experiment.status.value}]")
            print(f"  benchmark: {experiment.benchmark_name} v{experiment.benchmark_spec_version}")
            print(f"  dataset:   {experiment.dataset_version}")
            print(f"  started:   {experiment.start_time}")
            print(f"  ended:     {experiment.end_time}")
            print(f"  results ({len(results)}):")
            for r in results:
                status = "FAILED" if r.failed else f"WER={r.wer:.2f} CER={r.cer:.2f}"
                print(f"    [{r.model}] {r.test_case_id}: {status}")
        else:
            experiments = list_experiments(session)
            if not experiments:
                print("No experiments recorded yet. Run scripts/run_benchmark.py first.")
                return
            for e in experiments:
                print(f"{e.experiment_id}  [{e.status.value}]  {e.benchmark_name} @ {e.start_time}")


if __name__ == "__main__":
    main()
