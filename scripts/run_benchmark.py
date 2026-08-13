#!/usr/bin/env python
"""Command-line STT benchmark runner.

Persists every run to the database (build order step 11) via the
experiment/result repositories, and also writes a JSON report to
artifacts/reports/ for quick inspection without a DB client.

Usage:
    uv run python scripts/run_benchmark.py
    uv run python scripts/run_benchmark.py --dataset tests/fixtures/tiny_dataset.json \
        --models faster_whisper_tiny vosk_small_en
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from voice_benchmark.core.exceptions import InferenceError, ModelLoadError, ModelNotFoundError
from voice_benchmark.core.models import Experiment, ExperimentStatus, STTResult
from voice_benchmark.core.registry import get_stt_model, list_stt_models
from voice_benchmark.datasets.loader import load_dataset
from voice_benchmark.evaluation.stt_metrics import MetricComputationError, compute_stt_metrics
from voice_benchmark.storage.db import init_db, session_scope
from voice_benchmark.storage.repositories.experiment_repository import save_experiment
from voice_benchmark.storage.repositories.result_repository import save_stt_result


def run(dataset_path: str, model_names: list[str], output_dir: str) -> list[dict]:
    manifest = load_dataset(dataset_path)
    experiment_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    init_db()

    experiment = Experiment(
        experiment_id=experiment_id,
        benchmark_name=manifest.dataset.name,
        benchmark_spec_version="1.0",
        dataset_version=manifest.dataset.version,
        conditions=sorted({tc.condition for tc in manifest.test_cases}, key=lambda c: c.value),
        model_versions={name: None for name in model_names},
    )
    with session_scope() as session:
        save_experiment(session, experiment)

    all_results: list[dict] = []
    any_failed = False

    for model_name in model_names:
        try:
            adapter = get_stt_model(model_name)
        except ModelNotFoundError as exc:
            print(f"[skip] {exc}")
            continue

        print(f"\n=== {model_name} ===")

        try:
            adapter.load()
        except ModelLoadError as exc:
            print(f"[FAILED to load {model_name}] {exc}")
            any_failed = True
            continue

        for tc in manifest.test_cases:
            try:
                out = adapter.transcribe(tc.audio_path, language=tc.language)
                metrics = compute_stt_metrics(tc.reference_text, out["text"])
                result = STTResult(
                    experiment_id=experiment_id,
                    test_case_id=tc.id,
                    model=model_name,
                    prediction=out["text"],
                    reference=tc.reference_text,
                    latency_ms=out["latency_ms"],
                    audio_duration_sec=out["audio_duration_sec"],
                    rtf=out["rtf"],
                    wer=metrics["wer"],
                    cer=metrics["cer"],
                    condition=tc.condition,
                )
                print(
                    f"  {tc.id}: WER={result.wer:.2f} CER={result.cer:.2f} "
                    f"RTF={result.rtf:.3f} latency={result.latency_ms:.0f}ms"
                )
                print(f"    ref:  {tc.reference_text}")
                print(f"    pred: {out['text']}")
            except (InferenceError, MetricComputationError) as exc:
                result = STTResult(
                    experiment_id=experiment_id,
                    test_case_id=tc.id,
                    model=model_name,
                    reference=tc.reference_text,
                    condition=tc.condition,
                    failed=True,
                    error_message=str(exc),
                )
                print(f"  {tc.id}: FAILED -- {exc}")
                any_failed = True

            with session_scope() as session:
                save_stt_result(session, result)
            all_results.append(result.model_dump(mode="json"))

        adapter.unload()

    experiment.end_time = datetime.now(timezone.utc)
    experiment.status = ExperimentStatus.PARTIAL if any_failed else ExperimentStatus.SUCCESS
    with session_scope() as session:
        save_experiment(session, experiment)

    out_path = Path(output_dir) / f"{experiment_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nExperiment '{experiment_id}' status: {experiment.status.value}")
    print(f"Results written to {out_path} and to the database")

    return all_results


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
