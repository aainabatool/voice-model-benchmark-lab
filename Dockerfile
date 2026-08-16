# Dockerfile for containerized deployment of the API (e.g. any Docker-based
# host: Render, Fly.io, Hugging Face Spaces' Docker SDK, a VPS, etc).
# Listens on $PORT, defaulting to 7860.
#
# Local dev doesn't use this file at all -- it's only for hosted deployment.
# Local dev still uses `uv sync` / `uv run` directly, see README.

FROM python:3.12-slim

# System libs needed by the audio/TTS stack:
# - libsndfile1: required by soundfile (used throughout for reading/writing wav)
# - espeak-ng: backing engine for pyttsx3 TTS on Linux
# - libgomp1: OpenMP runtime needed by ctranslate2 (faster-whisper) and numba
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    espeak-ng \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy dependency files first so Docker can cache this layer across builds
# that only change application code, not dependencies.
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uv run uvicorn apps.api.main:app --host 0.0.0.0 --port ${PORT}"]
