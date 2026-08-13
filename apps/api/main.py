"""FastAPI application entrypoint.

Run with: uv run uvicorn apps.api.main:app --reload
Then open http://127.0.0.1:8000/docs for interactive API docs.
"""
from __future__ import annotations

from fastapi import FastAPI

from apps.api.routes import experiments, health, models

app = FastAPI(
    title="Voice Model Benchmark Lab API",
    description="HTTP API for running and inspecting STT/TTS model benchmarks.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(experiments.router)
