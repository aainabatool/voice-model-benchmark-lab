"""Model registry (STT and TTS).

A single, shared place mapping a model name to how to build its adapter
(spec section 10). This replaces any inline per-caller model dict -- the
CLI runner, the API, and the dashboard all import from here rather than
each hardcoding their own model list.

Adding a new model = adding one register_stt_model()/register_tts_model()
call. Nothing else in the codebase needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from voice_benchmark.adapters.base import STTModel, TTSModel
from voice_benchmark.adapters.stt.faster_whisper import FasterWhisperAdapter
from voice_benchmark.adapters.stt.vosk_adapter import VoskAdapter
from voice_benchmark.adapters.tts.pyttsx3_adapter import Pyttsx3Adapter
from voice_benchmark.core.exceptions import ModelNotFoundError


@dataclass(frozen=True)
class STTModelEntry:
    name: str
    factory: Callable[[], STTModel]
    description: str = ""


@dataclass(frozen=True)
class TTSModelEntry:
    name: str
    factory: Callable[[], TTSModel]
    description: str = ""


_STT_REGISTRY: dict[str, STTModelEntry] = {}
_TTS_REGISTRY: dict[str, TTSModelEntry] = {}


def register_stt_model(entry: STTModelEntry) -> None:
    _STT_REGISTRY[entry.name] = entry


def get_stt_model(name: str) -> STTModel:
    """Build a fresh adapter instance for `name`. Raises ModelNotFoundError
    (not KeyError) so callers get a consistent, catchable exception type."""
    if name not in _STT_REGISTRY:
        raise ModelNotFoundError(
            f"Unknown STT model '{name}'. Registered models: {list_stt_models()}"
        )
    return _STT_REGISTRY[name].factory()


def list_stt_models() -> list[str]:
    return list(_STT_REGISTRY)


def describe_stt_models() -> dict[str, str]:
    return {name: entry.description for name, entry in _STT_REGISTRY.items()}


def register_tts_model(entry: TTSModelEntry) -> None:
    _TTS_REGISTRY[entry.name] = entry


def get_tts_model(name: str) -> TTSModel:
    if name not in _TTS_REGISTRY:
        raise ModelNotFoundError(
            f"Unknown TTS model '{name}'. Registered models: {list_tts_models()}"
        )
    return _TTS_REGISTRY[name].factory()


def list_tts_models() -> list[str]:
    return list(_TTS_REGISTRY)


def describe_tts_models() -> dict[str, str]:
    return {name: entry.description for name, entry in _TTS_REGISTRY.items()}


# --------------------------------------------------------------------------
# Default registrations
# --------------------------------------------------------------------------

register_stt_model(
    STTModelEntry(
        name="faster_whisper_tiny",
        factory=lambda: FasterWhisperAdapter(model_size="tiny"),
        description="Faster-Whisper (CTranslate2), tiny size (~75MB)",
    )
)
register_stt_model(
    STTModelEntry(
        name="faster_whisper_base",
        factory=lambda: FasterWhisperAdapter(model_size="base"),
        description="Faster-Whisper (CTranslate2), base size (~150MB)",
    )
)
register_stt_model(
    STTModelEntry(
        name="faster_whisper_small",
        factory=lambda: FasterWhisperAdapter(model_size="small"),
        description="Faster-Whisper (CTranslate2), small size (~500MB)",
    )
)
register_stt_model(
    STTModelEntry(
        name="vosk_small_en",
        factory=lambda: VoskAdapter(model_name="vosk-model-small-en-us-0.15"),
        description="Vosk (Kaldi-based), small English model (~40MB), fully offline",
    )
)

register_tts_model(
    TTSModelEntry(
        name="pyttsx3",
        factory=lambda: Pyttsx3Adapter(rate=175),
        description="pyttsx3 (OS-native TTS: SAPI5/espeak/NSSpeechSynthesizer), no download needed",
    )
)
