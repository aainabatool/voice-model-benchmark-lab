"""Endpoint for listing registered STT models."""
from __future__ import annotations

from fastapi import APIRouter

from voice_benchmark.core.registry import describe_stt_models

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models() -> dict[str, str]:
    """Model name -> human-readable description, from the registry."""
    return describe_stt_models()
