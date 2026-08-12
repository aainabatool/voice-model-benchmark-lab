"""Timing helpers.

Every benchmark run needs latency in milliseconds, consistently measured.
This wraps time.perf_counter() (monotonic, high-resolution -- not
time.time(), which can jump around) in a small context manager.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Timer:
    """Usage:

        with Timer() as t:
            do_work()
        print(t.elapsed_ms)
    """

    elapsed_ms: float = 0.0
    _start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def compute_rtf(processing_time_sec: float, audio_duration_sec: float) -> float | None:
    """Real-time factor: how many seconds of compute per second of audio.

    RTF < 1.0 means faster than real time. Returns None if audio_duration_sec
    is zero/invalid rather than raising ZeroDivisionError -- a malformed
    audio file shouldn't crash the whole benchmark run.
    """
    if not audio_duration_sec or audio_duration_sec <= 0:
        return None
    return processing_time_sec / audio_duration_sec
