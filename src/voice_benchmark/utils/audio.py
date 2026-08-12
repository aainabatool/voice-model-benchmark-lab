"""Audio utility functions.

Thin wrappers around soundfile/librosa so the rest of the codebase never
touches those libraries directly -- if we swap audio backends later, only
this file changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


class AudioLoadError(RuntimeError):
    """Raised when an audio file can't be read."""


def get_audio_duration_sec(audio_path: str | Path) -> float:
    """Duration in seconds without loading the full waveform into memory."""
    path = Path(audio_path)
    if not path.exists():
        raise AudioLoadError(f"Audio file not found: {path}")
    try:
        info = sf.info(str(path))
    except Exception as exc:  # noqa: BLE001 -- want a consistent AudioLoadError type
        raise AudioLoadError(f"Could not read audio file {path}: {exc}") from exc
    return info.frames / info.samplerate


def load_audio(
    audio_path: str | Path,
    target_sr: int | None = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load an audio file as a float32 waveform.

    Returns (samples, sample_rate). If target_sr is given and differs from
    the file's native rate, resamples using soxr (via librosa) -- most STT
    models expect 16kHz mono regardless of the source file's format.
    """
    path = Path(audio_path)
    if not path.exists():
        raise AudioLoadError(f"Audio file not found: {path}")

    try:
        samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        raise AudioLoadError(f"Could not read audio file {path}: {exc}") from exc

    if mono and samples.ndim > 1:
        samples = samples.mean(axis=1)

    if target_sr is not None and sr != target_sr:
        import librosa  # local import: librosa is heavier, only pull it in when resampling

        samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return samples.astype(np.float32), sr


def compute_silence_ratio(samples: np.ndarray, sr: int, top_db: int = 30) -> float:
    """Fraction of the signal that librosa considers silence.

    Used for TTS objective metrics (spec section 12) -- flags outputs that
    are mostly silence, which a naive duration/latency metric wouldn't catch.
    """
    if samples.size == 0:
        return 1.0
    import librosa

    intervals = librosa.effects.split(samples, top_db=top_db)
    voiced_samples = sum(end - start for start, end in intervals)
    return 1.0 - (voiced_samples / samples.size)
