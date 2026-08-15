#!/usr/bin/env python
"""Generate noisy variants of the STT fixture audio for robustness testing.

Reads the clean fixture dataset, adds white Gaussian noise at each of the
standard SNR conditions (spec section 13), writes the noisy wav files,
and produces a combined manifest (clean baseline + all noise levels)
that scripts/run_benchmark.py can run against directly.

No orchestration/API/DB changes were needed to support this -- `condition`
has been threaded through TestCase/STTResult since build order step 8, so
a benchmark run against this manifest automatically breaks results down
by noise level.

Usage:
    uv run python scripts/generate_noisy_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

from voice_benchmark.core.models import Condition
from voice_benchmark.datasets.loader import load_dataset
from voice_benchmark.utils.audio import load_audio
from voice_benchmark.utils.noise import add_noise_at_snr, measure_snr_db

SOURCE_DATASET = "tests/fixtures/tiny_dataset.json"
OUTPUT_DATASET = "tests/fixtures/tiny_noise_dataset.json"
NOISY_AUDIO_DIR = Path("tests/fixtures/audio/noisy")

CONDITIONS_DB: dict[Condition, float] = {
    Condition.NOISE_20DB: 20.0,
    Condition.NOISE_10DB: 10.0,
    Condition.NOISE_5DB: 5.0,
    Condition.NOISE_0DB: 0.0,
    Condition.NOISE_NEG5DB: -5.0,
}


def main() -> None:
    manifest = load_dataset(SOURCE_DATASET)

    test_cases: list[dict] = []
    for tc in manifest.test_cases:
        # Keep the clean version too -- it's the baseline point on the
        # robustness curve (effectively infinite SNR).
        test_cases.append(tc.model_dump(mode="json"))

        clean_samples, sr = load_audio(tc.audio_path)

        for condition, snr_db in CONDITIONS_DB.items():
            seed = abs(hash((tc.id, condition.value))) % (2**31)
            noisy = add_noise_at_snr(clean_samples, snr_db, seed=seed)
            achieved_snr = measure_snr_db(clean_samples, noisy)

            out_dir = NOISY_AUDIO_DIR / condition.value
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{tc.id}.wav"
            sf.write(str(out_path), noisy, sr)

            print(
                f"{tc.id} @ {condition.value}: target={snr_db:>5.1f}dB "
                f"achieved={achieved_snr:6.2f}dB -> {out_path}"
            )

            new_tc = tc.model_dump(mode="json")
            new_tc["id"] = f"{tc.id}_{condition.value}"
            new_tc["audio_path"] = str(out_path)
            new_tc["condition"] = condition.value
            test_cases.append(new_tc)

    output_manifest = {
        "dataset": {
            "name": "tiny_noise_fixture",
            "version": "1.0",
            "language": manifest.dataset.language,
        },
        "test_cases": test_cases,
    }

    Path(OUTPUT_DATASET).write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(test_cases)} test cases ({len(manifest.test_cases)} clips x "
          f"{1 + len(CONDITIONS_DB)} conditions) to {OUTPUT_DATASET}")


if __name__ == "__main__":
    main()
