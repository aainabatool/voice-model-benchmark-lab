"""FastAPI application entrypoint.

Run with: uv run uvicorn apps.api.main:app --reload
Then open http://127.0.0.1:8000/docs for interactive API docs.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routes import experiments, health, models, tts_experiments

app = FastAPI(
    title="Voice Model Benchmark Lab API",
    description="HTTP API for running and inspecting STT/TTS model benchmarks.",
    version="0.1.0",
)

# Open CORS: this is a public read-mostly demo API with no auth or sensitive
# data, and the dashboard is meant to be deployed on a different domain
# (Streamlit Community Cloud) than the API (e.g. Render) -- without this,
# every request from the deployed dashboard would be blocked by the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(experiments.router)
app.include_router(tts_experiments.router)
