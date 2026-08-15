"""Noise injection for robustness testing.

Adds white Gaussian noise to a clean waveform at a target SNR (spec
section 13: noise/robustness benchmarks). SNR is measured on average
signal power -- the standard definition for audio SNR.
"""
from __future__ import annotations

import numpy as np


def add_noise_at_snr(
    samples: np.ndarray, snr_db: float, seed: int | None = None
) -> np.ndarray:
    """Return a copy of `samples` with white Gaussian noise added at the
    given SNR (in dB). Positive SNR = noise quieter than signal; negative
    SNR = noise louder than signal.
    """
    rng = np.random.default_rng(seed)

    signal_power = float(np.mean(samples.astype(np.float64) ** 2))
    if signal_power <= 0:
        # Silent input -- nothing meaningful to scale noise against.
        # Use a small fixed noise floor rather than raising, since this
        # is a legitimate (if unusual) edge case, not an error.
        signal_power = 1e-6

    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=samples.shape)

    mixed = samples.astype(np.float64) + noise

    # Peak-normalize if this would clip -- scales signal+noise together,
    # so the SNR ratio itself is preserved; only headroom is added.
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak

    return mixed.astype(np.float32)


def measure_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Measure the actual achieved SNR between a clean reference and its
    noisy counterpart. Used to verify add_noise_at_snr hit its target,
    since peak-normalization can shift the result slightly."""
    noise = noisy.astype(np.float64) - clean.astype(np.float64)
    signal_power = float(np.mean(clean.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise**2))
    if noise_power <= 0:
        return float("inf")
    return 10.0 * float(np.log10(signal_power / noise_power))
