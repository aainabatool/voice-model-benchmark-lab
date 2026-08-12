# Voice Model Benchmark Lab

A local-first benchmarking platform for speech and voice models (STT + TTS).
Instead of picking a model by reputation or a single demo, it runs the same
benchmark specification against multiple models and stores comparable,
reproducible results.

**Central question:** Which voice model is best for this task, language,
hardware environment, and audio condition?

## Status

Milestone 1 (MVP) in progress -- see Build Order below.

## Initial scope

- Speech-to-Text (STT) benchmarking
- Text-to-Speech (TTS) benchmarking
- Noise and robustness testing
- Latency and real-time performance measurement
- Objective quality and accuracy metrics
- Experiment tracking and reproducibility
- Interactive dashboard for comparison and analysis

## Stack

Python 3.12, uv, FastAPI, Streamlit, SQLAlchemy (SQLite for local dev,
PostgreSQL for staging/prod), jiwer, librosa

## Quickstart

```bash
uv sync
cp .env.example .env
uv run uvicorn apps.api.main:app --reload
uv run streamlit run apps/dashboard/app.py
```

## Project layout
## Build order

Following the project spec's recommended order -- no TTS or robustness until
STT is solid:

1. [x] Initialize repo with uv
2. [x] Create src package and basic configuration
3. [ ] Audio utility functions
4. [ ] BenchmarkSpec, TestCase, ModelMetadata, Result schemas
5. [ ] One STT adapter (Faster-Whisper)
6. [ ] WER/CER + timing
7. [ ] Tiny fixture dataset
8. [ ] Command-line benchmark runner
9. [ ] Second STT adapter
10. [ ] Model registry
11. [ ] Experiment persistence
12. [ ] FastAPI endpoints
13. [ ] Streamlit dashboard
14. [ ] Hardware metadata
15. [ ] TTS adapters
16. [ ] Robustness / noise generation
17. [ ] Analytics + Pareto visualization
18. [ ] Reports / export
19. [ ] Voice-agent benchmarking (V2/V3)

## License

TBD
