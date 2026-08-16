"""TTS experiment endpoints: list, get detail, and trigger new runs.

Mirrors experiments.py's STT design, but for TTS. STT and TTS
experiments share the same `experiments` table (no type flag needed) --
an experiment is "a TTS experiment" simply if it has tts_results rows
rather than stt_results rows, which is how list_tts_experiments below
finds them.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from voice_benchmark.core.exceptions import DatasetError
from voice_benchmark.core.models import Experiment, TTSResult
from voice_benchmark.orchestration.runner import new_experiment_id, run_tts_benchmark
from voice_benchmark.storage.db import init_db, session_scope
from voice_benchmark.storage.models import TTSResultORM
from voice_benchmark.storage.repositories.experiment_repository import (
    get_experiment,
    save_experiment,
)
from voice_benchmark.storage.repositories.tts_result_repository import (
    get_tts_results_for_experiment,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tts-experiments", tags=["tts-experiments"])


class TTSRunRequest(BaseModel):
    dataset_path: str = "tests/fixtures/tiny_tts_dataset.json"
    models: list[str]
    output_dir: str = "artifacts/audio"


class TTSRunAccepted(BaseModel):
    experiment_id: str
    message: str = "TTS run started. Poll GET /tts-experiments/{experiment_id} for status."


class TTSExperimentDetail(BaseModel):
    experiment: Experiment
    results: list[TTSResult]


def _run_in_background(
    experiment_id: str, dataset_path: str, model_names: list[str], output_dir: str
) -> None:
    try:
        run_tts_benchmark(experiment_id, dataset_path, model_names, output_dir)
    except DatasetError as exc:
        logger.error("TTS run '%s' failed: %s", experiment_id, exc)


@router.post("", response_model=TTSRunAccepted, status_code=202)
def start_tts_run(request: TTSRunRequest, background_tasks: BackgroundTasks) -> TTSRunAccepted:
    """Kick off a TTS run in the background. Returns immediately with the
    experiment_id -- poll GET /tts-experiments/{experiment_id} for status."""
    experiment_id = new_experiment_id()

    init_db()
    placeholder = Experiment(
        experiment_id=experiment_id,
        benchmark_name="(starting)",
        benchmark_spec_version="1.0",
        dataset_version="(starting)",
        conditions=[],
    )
    with session_scope() as session:
        save_experiment(session, placeholder)

    background_tasks.add_task(
        _run_in_background, experiment_id, request.dataset_path, request.models, request.output_dir
    )
    return TTSRunAccepted(experiment_id=experiment_id)


@router.get("", response_model=list[Experiment])
def list_tts_experiments() -> list[Experiment]:
    """Experiments that have at least one TTS result recorded."""
    init_db()
    with session_scope() as session:
        tts_experiment_ids = {
            row[0] for row in session.query(TTSResultORM.experiment_id).distinct().all()
        }
        experiments = [get_experiment(session, eid) for eid in tts_experiment_ids]
        experiments = [e for e in experiments if e is not None]
    return sorted(experiments, key=lambda e: e.start_time, reverse=True)


@router.get("/{experiment_id}", response_model=TTSExperimentDetail)
def get_tts_experiment_detail(experiment_id: str) -> TTSExperimentDetail:
    init_db()
    with session_scope() as session:
        experiment = get_experiment(session, experiment_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
        results = get_tts_results_for_experiment(session, experiment_id)
    return TTSExperimentDetail(experiment=experiment, results=results)
