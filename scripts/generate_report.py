#!/usr/bin/env python
"""Generate a Markdown report for one experiment.

Usage:
    uv run python scripts/generate_report.py --experiment-id run_...
    uv run python scripts/generate_report.py --experiment-id run_... --stdout
"""
from __future__ import annotations

import argparse
from pathlib import Path

from voice_benchmark.reports.report_generator import generate_markdown_report
from voice_benchmark.storage.db import init_db, session_scope
from voice_benchmark.storage.repositories.experiment_repository import get_experiment
from voice_benchmark.storage.repositories.result_repository import get_results_for_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown report for an experiment.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", default="artifacts/reports")
    parser.add_argument(
        "--stdout", action="store_true", help="Print the report instead of writing a file."
    )
    args = parser.parse_args()

    init_db()
    with session_scope() as session:
        experiment = get_experiment(session, args.experiment_id)
        if experiment is None:
            print(f"No experiment found with id '{args.experiment_id}'")
            return
        results = get_results_for_experiment(session, args.experiment_id)

    report = generate_markdown_report(experiment, results)

    if args.stdout:
        print(report)
        return

    out_path = Path(args.output) / f"{args.experiment_id}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
