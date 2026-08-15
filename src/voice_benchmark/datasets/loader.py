"""Dataset loading.

Reads a manifest JSON file, validates it against DatasetManifest, and
checks that every referenced audio file actually exists -- fail fast at
load time rather than partway through a benchmark run.
"""
from __future__ import annotations

import json
from pathlib import Path

from voice_benchmark.core.exceptions import DatasetError
from voice_benchmark.datasets.schemas import DatasetManifest


def load_dataset(manifest_path: str | Path) -> DatasetManifest:
    path = Path(manifest_path)
    if not path.exists():
        raise DatasetError(f"Dataset manifest not found: {path}")

    try:
        # utf-8-sig transparently strips a BOM if present (e.g. from
        # PowerShell's `Set-Content -Encoding utf8`, which writes UTF-8
        # WITH a BOM despite the name) and behaves identically to utf-8
        # for files that don't have one.
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        manifest = DatasetManifest.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 -- normalize any parse/validation error
        raise DatasetError(f"Invalid dataset manifest {path}: {exc}") from exc

    missing = [
        tc.id
        for tc in manifest.test_cases
        if tc.audio_path and not Path(tc.audio_path).exists()
    ]
    if missing:
        raise DatasetError(
            f"Dataset '{manifest.dataset.name}': missing audio files for test cases: {missing}"
        )

    return manifest
