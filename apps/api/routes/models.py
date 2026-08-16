"""Endpoints for listing registered STT and TTS models."""
from __future__ import annotations

from fastapi import APIRouter

from voice_benchmark.core.registry import describe_stt_models, describe_tts_models

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models() -> dict[str, str]:
    """STT model name -> human-readable description, from the registry."""
    return describe_stt_models()


@router.get("/tts")
def list_tts_models_endpoint() -> dict[str, str]:
    """TTS model name -> human-readable description, from the registry."""
    return describe_tts_models()
