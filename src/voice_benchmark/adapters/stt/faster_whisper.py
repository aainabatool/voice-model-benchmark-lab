"""Faster-Whisper STT adapter.

Wraps the faster-whisper library behind the common STTModel interface.
Model weights are downloaded from Hugging Face on first load() and cached
locally by the library -- that first call needs internet access and will
be slower than subsequent runs.
"""
from __future__ import annotations

from typing import Any

from voice_benchmark.adapters.base import STTModel
from voice_benchmark.core.exceptions import InferenceError, ModelLoadError
from voice_benchmark.utils.audio import AudioLoadError, get_audio_duration_sec
from voice_benchmark.utils.timing import Timer, compute_rtf


class FasterWhisperAdapter(STTModel):
    """Adapter for https://github.com/SYSTRAN/faster-whisper.

    model_size: any faster-whisper model name, e.g. "tiny", "base", "small",
    "medium", "large-v3". Smaller = faster to download and run, useful for
    first getting the pipeline working end to end.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
    ) -> None:
        self.name = f"faster_whisper_{model_size}"
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ModelLoadError(
                "faster-whisper is not installed. Run: uv add faster-whisper"
            ) from exc

        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:  # noqa: BLE001 -- normalize all load failures to ModelLoadError
            raise ModelLoadError(
                f"Failed to load faster-whisper model '{self.model_size}': {exc}"
            ) from exc

    def transcribe(self, audio_path: str, language: str | None = None) -> dict[str, Any]:
        if self._model is None:
            raise ModelLoadError(f"{self.name}: model not loaded -- call load() first")

        try:
            audio_duration_sec = get_audio_duration_sec(audio_path)
        except AudioLoadError as exc:
            raise InferenceError(str(exc), model=self.name) from exc

        try:
            with Timer() as timer:
                segments, info = self._model.transcribe(audio_path, language=language)
                text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:  # noqa: BLE001 -- normalize all inference failures
            raise InferenceError(
                f"faster-whisper transcription failed: {exc}", model=self.name
            ) from exc

        rtf = compute_rtf(timer.elapsed_ms / 1000.0, audio_duration_sec)

        return {
            "text": text,
            "latency_ms": timer.elapsed_ms,
            "audio_duration_sec": audio_duration_sec,
            "rtf": rtf,
            "detected_language": getattr(info, "language", language),
            "model_metadata": {
                "name": self.name,
                "adapter": "faster_whisper",
                "config": {
                    "model_size": self.model_size,
                    "device": self.device,
                    "compute_type": self.compute_type,
                },
            },
        }

    def unload(self) -> None:
        self._model = None
