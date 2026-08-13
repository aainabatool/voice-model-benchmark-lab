#!/usr/bin/env python
"""Command-line STT benchmark runner.

This is intentionally simple -- no database persistence yet (that's build
order step 11). Results print to the console as a run log and are also
written as JSON to artifacts/reports/.

Usage:
    uv run python scripts/run_benchmark.py
    uv run python scripts/run_benchmark.py --dataset tests/fixtures/tiny_dataset.json \
        --models faster_whisper_tiny faster_whisper_base
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from voice_benchmark.adapters.stt.faster_whisper import FasterWhisperAdapter
from voice_benchmark.core.exceptions import InferenceError, ModelLoadError
from voice_benchmark.core.models import STTResult
from voice_benchmark.datasets.loader import load_dataset
from voice_benchmark.evaluation.stt_metrics import MetricComputationError, compute_stt_metrics

# Known model names -> adapter factory. Extend this as more STT adapters
# are added (build order step 9 onward covers a real model registry).
_MODEL_FACTORIES = {
    "faster_whisper_tiny": lambda: FasterWhisperAdapter(model_size="tiny"),
    "faster_whisper_base": lambda: FasterWhisperAdapter(model_size="base"),
    "faster_whisper_small": lambda: FasterWhisperAdapter(model_size="small"),
}


def run(dataset_path: str, model_names: list[str], output_dir: str) -> list[dict]:
    manifest = load_dataset(dataset_path)
    experiment_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    all_results: list[dict] = []

    for model_name in model_names:
        if model_name not in _MODEL_FACTORIES:
            print(f"[skip] Unknown model '{model_name}'. Known: {list(_MODEL_FACTORIES)}")
            continue

        print(f"\n=== {model_name} ===")
        adapter = _MODEL_FACTORIES[model_name]()

        try:
            adapter.load()
        except ModelLoadError as exc:
            print(f"[FAILED to load {model_name}] {exc}")
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

            all_results.append(result.model_dump(mode="json"))

        adapter.unload()

    out_path = Path(output_dir) / f"{experiment_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an STT benchmark from the command line.")
    parser.add_argument("--dataset", default="tests/fixtures/tiny_dataset.json")
    parser.add_argument(
        "--models", nargs="+", default=["faster_whisper_tiny", "faster_whisper_base"]
    )
    parser.add_argument("--output", default="artifacts/reports")
    args = parser.parse_args()

    run(args.dataset, args.models, args.output)


if __name__ == "__main__":
    main()
