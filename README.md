# Voice Model Benchmark Lab

A local-first benchmarking platform for speech and voice models (STT + TTS).
Instead of picking a model by reputation or a single demo, it runs the same
benchmark specification against multiple models and stores comparable,
reproducible results.

**Central question:** Which voice model is best for this task, language,
hardware environment, and audio condition?

## What it does

- Runs multiple STT models (Faster-Whisper, Vosk) against the same dataset
  and computes WER/CER, latency, and real-time factor (RTF)
- Runs TTS models (pyttsx3) and computes objective quality metrics
  (RTF, speech rate, silence ratio)
- Injects controlled white-noise conditions (20dB down to -5dB SNR) to
  measure robustness, not just clean-audio accuracy
- Records hardware (CPU/RAM/GPU) with every run, since RTF numbers are
  meaningless without knowing what they were measured on
- Persists every experiment to a database, queryable via CLI, a REST API,
  or a Streamlit dashboard
- Computes the Pareto frontier (accuracy vs. speed) so "best" isn't a
  single number when it's really a tradeoff
- Generates a shareable Markdown report per experiment

## Stack

Python 3.12 &middot; [uv](https://docs.astral.sh/uv/) &middot; FastAPI &middot;
Streamlit &middot; SQLAlchemy (SQLite for local dev, Postgres-ready) &middot;
Faster-Whisper &middot; Vosk &middot; pyttsx3 &middot; jiwer &middot; librosa

## Quickstart

```bash
uv sync
uv run uvicorn apps.api.main:app --reload      # terminal 1: API on :8000
uv run streamlit run apps/dashboard/app.py     # terminal 2: dashboard on :8501
```

Or skip the API/dashboard and just use the CLI:

```bash
uv run python scripts/run_benchmark.py --models faster_whisper_tiny vosk_small_en
uv run python scripts/generate_noisy_fixtures.py
uv run python scripts/run_benchmark.py --dataset tests/fixtures/tiny_noise_dataset.json
uv run python scripts/analyze_experiment.py --experiment-id <run_id>
uv run python scripts/generate_report.py --experiment-id <run_id> --stdout
uv run python scripts/run_tts_benchmark.py --models pyttsx3
```

## Project layout
## Deployment

The dashboard and API are deployed separately: Streamlit Community Cloud
hosts the dashboard, and it talks over HTTP to an API hosted elsewhere
(e.g. Render). They don't need to be on the same platform.

### API on Render

1. Push this repo to GitHub (already done if you're reading this there).
2. In Render: **New > Blueprint**, point it at this repo. Render reads
   `render.yaml` at the repo root automatically.
3. Deploy. Note the public URL Render gives you, e.g.
   `https://voice-benchmark-api.onrender.com`.

Free-tier caveats (see comments in `render.yaml` for detail): limited RAM,
the instance spins down when idle, and SQLite data doesn't persist across
restarts on the free plan unless you point `DATABASE_URL` at a hosted
Postgres instance instead.

### Dashboard on Streamlit Community Cloud

1. In [Streamlit Community Cloud](https://share.streamlit.io): **New app**,
   point it at this repo, set the main file path to `apps/dashboard/app.py`.
   Streamlit Cloud automatically finds `apps/dashboard/requirements.txt`
   (a small dashboard-only dependency list, separate from the full
   project's `pyproject.toml`).
2. In the app's **Settings > Secrets**, add:
```toml
   API_BASE_URL = "https://voice-benchmark-api.onrender.com"
```
   This makes the dashboard default to your deployed API instead of
   `localhost` (the sidebar field is still editable at runtime either way).
3. Deploy.

## Build order

All core steps from the original spec are complete:

1. [x] Initialize repo with uv
2. [x] Create src package and basic configuration
3. [x] Audio utility functions
4. [x] BenchmarkSpec, TestCase, ModelMetadata, Result schemas
5. [x] One STT adapter (Faster-Whisper)
6. [x] WER/CER + timing
7. [x] Tiny fixture dataset (real speech, synthesized via OS TTS)
8. [x] Command-line benchmark runner
9. [x] Second STT adapter (Vosk -- a genuinely different engine)
10. [x] Model registry
11. [x] Experiment persistence (SQLAlchemy)
12. [x] FastAPI endpoints
13. [x] Streamlit dashboard
14. [x] Hardware metadata
15. [x] TTS adapter (pyttsx3) + objective metrics
16. [x] Noise injection + robustness testing
17. [x] Pareto frontier analysis
18. [x] Markdown report generation

Not started (V2/V3 stretch, out of core scope): voice-agent benchmarking.

## License

TBD
