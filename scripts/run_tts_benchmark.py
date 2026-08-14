#!/usr/bin/env python
"""Command-line TTS benchmark runner.

Synthesizes each test case's reference_text as a prompt, computes
objective TTS metrics (RTF, speech rate, silence ratio -- spec section 12),
and persists to the database via the shared orchestration runner.

Usage:
    uv run python scripts/run_tts_benchmark.py
    uv run python scripts/run_tts_benchmark.py --models pyttsx3
"""
from __future__ import annotations

import argparse

from voice_benchmark.core.models import TTSResult
from voice_benchmark.core.registry import list_tts_models
from voice_benchmark.orchestration.runner import new_experiment_id, run_tts_benchmark


def _print_model_status(model_name: str, status: str) -> None:
    print(f"\n=== {model_name} === ({status})")


def _print_result(model_name: str, result: TTSResult) -> None:
    if result.failed:
        print(f"  {result.test_case_id}: FAILED -- {result.error_message}")
        return
    print(
        f"  {result.test_case_id}: RTF={result.rtf:.3f} "
        f"duration={result.output_duration_sec:.2f}s "
        f"speech_rate={result.speech_rate_wpm:.0f}wpm "
        f"silence_ratio={result.silence_ratio:.2f}"
    )
    print(f"    output: {result.output_path}")


def run(dataset_path: str, model_names: list[str], output_dir: str) -> list[dict]:
    experiment_id = new_experiment_id()
    experiment, results = run_tts_benchmark(
        experiment_id,
        dataset_path,
        model_names,
        output_dir,
        on_result=_print_result,
        on_model_status=_print_model_status,
    )

    print(f"\nExperiment '{experiment_id}' status: {experiment.status.value}")
    if experiment.hardware:
        hw = experiment.hardware
        gpu = f"{hw.gpu} ({hw.vram_gb}GB VRAM)" if hw.gpu else "none (CPU only)"
        print(f"Hardware: {hw.os}, Python {hw.python_version}, {hw.cpu}, {hw.ram_gb}GB RAM, GPU: {gpu}")
    print(f"Audio written under {output_dir}/{experiment_id}/ and results saved to the database")

    return [r.model_dump(mode="json") for r in results]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TTS benchmark from the command line.")
    parser.add_argument("--dataset", default="tests/fixtures/tiny_tts_dataset.json")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pyttsx3"],
        help=f"Model names from the registry. Available: {list_tts_models()}",
    )
    parser.add_argument("--output", default="artifacts/audio")
    parser.add_argument(
        "--list-models", action="store_true", help="List registered TTS models and exit."
    )
    args = parser.parse_args()

    if args.list_models:
        for name in list_tts_models():
            print(name)
        return

    run(args.dataset, args.models, args.output)


if __name__ == "__main__":
    main()
