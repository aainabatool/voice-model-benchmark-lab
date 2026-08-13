#!/usr/bin/env python
"""Command-line STT benchmark runner.

Thin wrapper around voice_benchmark.orchestration.runner.run_stt_benchmark
-- prints progress to the console and writes a JSON report, while the
shared runner handles the actual model loop and DB persistence (also used
by the API, see apps/api/routes/experiments.py).

Usage:
    uv run python scripts/run_benchmark.py
    uv run python scripts/run_benchmark.py --dataset tests/fixtures/tiny_dataset.json \
        --models faster_whisper_tiny vosk_small_en
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from voice_benchmark.core.models import STTResult
from voice_benchmark.core.registry import list_stt_models
from voice_benchmark.orchestration.runner import new_experiment_id, run_stt_benchmark


def _print_model_status(model_name: str, status: str) -> None:
    print(f"\n=== {model_name} === ({status})")


def _print_result(model_name: str, result: STTResult) -> None:
    if result.failed:
        print(f"  {result.test_case_id}: FAILED -- {result.error_message}")
        return
    print(
        f"  {result.test_case_id}: WER={result.wer:.2f} CER={result.cer:.2f} "
        f"RTF={result.rtf:.3f} latency={result.latency_ms:.0f}ms"
    )
    print(f"    ref:  {result.reference}")
    print(f"    pred: {result.prediction}")


def run(dataset_path: str, model_names: list[str], output_dir: str) -> list[dict]:
    experiment_id = new_experiment_id()
    experiment, results = run_stt_benchmark(
        experiment_id,
        dataset_path,
        model_names,
        on_result=_print_result,
        on_model_status=_print_model_status,
    )

    out_path = Path(output_dir) / f"{experiment_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([r.model_dump(mode="json") for r in results], indent=2), encoding="utf-8"
    )
    print(f"\nExperiment '{experiment_id}' status: {experiment.status.value}")
    print(f"Results written to {out_path} and to the database")

    return [r.model_dump(mode="json") for r in results]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an STT benchmark from the command line.")
    parser.add_argument("--dataset", default="tests/fixtures/tiny_dataset.json")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["faster_whisper_tiny", "vosk_small_en"],
        help=f"Model names from the registry. Available: {list_stt_models()}",
    )
    parser.add_argument("--output", default="artifacts/reports")
    parser.add_argument(
        "--list-models", action="store_true", help="List registered models and exit."
    )
    args = parser.parse_args()

    if args.list_models:
        for name in list_stt_models():
            print(name)
        return

    run(args.dataset, args.models, args.output)


if __name__ == "__main__":
    main()
