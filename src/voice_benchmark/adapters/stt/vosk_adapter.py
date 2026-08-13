"""Vosk STT adapter.

Vosk (https://alphacephei.com/vosk/) is a lightweight, fully offline
speech recognizer built on Kaldi -- architecturally distinct from
Whisper/CTranslate2. Having a second, genuinely different engine (not
just another Whisper size) is the point of build order step 9: it's what
makes model comparisons meaningful rather than comparing a family against
itself.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from voice_benchmark.adapters.base import STTModel
from voice_benchmark.core.exceptions import InferenceError, ModelLoadError
from voice_benchmark.utils.audio import AudioLoadError, get_audio_duration_sec, load_audio
from voice_benchmark.utils.timing import Timer, compute_rtf

_VOSK_SAMPLE_RATE = 16000  # Vosk models are trained on 16kHz mono PCM
_CHUNK_SIZE = 4000  # feed audio in chunks, mirroring real streaming usage


class VoskAdapter(STTModel):
    """Adapter for https://github.com/alphacep/vosk-api.

    model_name: a Vosk model name from https://alphacephei.com/vosk/models
    (e.g. "vosk-model-small-en-us-0.15"). Downloaded and cached
    automatically by the vosk package on first load().
    """

    def __init__(self, model_name: str = "vosk-model-small-en-us-0.15") -> None:
        self.name = f"vosk_{model_name}"
        self.model_name = model_name
        self._model: Any = None

    def load(self) -> None:
        try:
            from vosk import Model
        except ImportError as exc:
            raise ModelLoadError("vosk is not installed. Run: uv add vosk") from exc

        try:
            self._model = Model(model_name=self.model_name)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Failed to load vosk model '{self.model_name}': {exc}"
            ) from exc

    def transcribe(self, audio_path: str, language: str | None = None) -> dict[str, Any]:
        if self._model is None:
            raise ModelLoadError(f"{self.name}: model not loaded -- call load() first")

        try:
            audio_duration_sec = get_audio_duration_sec(audio_path)
            samples, sr = load_audio(audio_path, target_sr=_VOSK_SAMPLE_RATE)
        except AudioLoadError as exc:
            raise InferenceError(str(exc), model=self.name) from exc

        try:
            from vosk import KaldiRecognizer

            with Timer() as timer:
                recognizer = KaldiRecognizer(self._model, sr)
                recognizer.SetWords(False)

                pcm16 = (samples * 32767).astype(np.int16).tobytes()
                for i in range(0, len(pcm16), _CHUNK_SIZE):
                    recognizer.AcceptWaveform(pcm16[i : i + _CHUNK_SIZE])

                final = json.loads(recognizer.FinalResult())
                text = final.get("text", "").strip()
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"vosk transcription failed: {exc}", model=self.name) from exc

        rtf = compute_rtf(timer.elapsed_ms / 1000.0, audio_duration_sec)

        return {
            "text": text,
            "latency_ms": timer.elapsed_ms,
            "audio_duration_sec": audio_duration_sec,
            "rtf": rtf,
            "detected_language": language,
            "model_metadata": {
                "name": self.name,
                "adapter": "vosk",
                "config": {"model_name": self.model_name},
            },
        }

    def unload(self) -> None:
        self._model = None
