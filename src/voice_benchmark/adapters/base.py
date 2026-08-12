"""Common model adapter interfaces.

The benchmark engine must never know how a specific model works internally
(spec section 10) -- it only calls these methods. Every model gets one
adapter file behind this interface; adding a new model means adding a new
adapter, never touching the benchmark engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class STTModel(ABC):
    """Speech-to-text model adapter."""

    name: str

    @abstractmethod
    def load(self) -> None:
        """Load model weights / initialize the runtime. Called once before
        any transcribe() calls."""

    @abstractmethod
    def transcribe(self, audio_path: str, language: str | None = None) -> dict[str, Any]:
        """Transcribe one audio file.

        Returns a dict with at least: text, latency_ms, audio_duration_sec,
        rtf. Raises voice_benchmark.core.exceptions.InferenceError on
        failure -- never returns a silently-empty/zero result for a failed
        run.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release model resources (GPU memory, etc.)."""


class TTSModel(ABC):
    """Text-to-speech model adapter."""

    name: str

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def synthesize(self, text: str, output_path: str, **kwargs: Any) -> dict[str, Any]:
        """Synthesize speech for `text`, writing audio to `output_path`.

        Returns a dict with at least: output_path, generation_latency_ms,
        output_duration_sec, rtf.
        """

    @abstractmethod
    def unload(self) -> None: ...
