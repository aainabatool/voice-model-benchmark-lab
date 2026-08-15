"""Experiment endpoints: list, get detail, and trigger new runs.

POST /experiments returns immediately with the experiment_id (status
"running", inserted before the background task starts) so a client can
poll GET /experiments/{experiment_id} right away rather than getting a
transient 404 while the run is still starting up.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from voice_benchmark.core.exceptions import DatasetError
from voice_benchmark.core.models import Experiment, STTResult
from voice_benchmark.orchestration.runner import new_experiment_id, run_stt_benchmark
from voice_benchmark.reports.report_generator import generate_markdown_report
from voice_benchmark.storage.db import init_db, session_scope
from voice_benchmark.storage.repositories.experiment_repository import (
    get_experiment,
    list_experiments,
    save_experiment,
)
from voice_benchmark.storage.repositories.result_repository import get_results_for_experiment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/experiments", tags=["experiments"])


class RunRequest(BaseModel):
    dataset_path: str = "tests/fixtures/tiny_dataset.json"
    models: list[str]


class RunAccepted(BaseModel):
    experiment_id: str
    message: str = "Benchmark run started. Poll GET /experiments/{experiment_id} for status."


class ExperimentDetail(BaseModel):
    experiment: Experiment
    results: list[STTResult]


def _run_in_background(experiment_id: str, dataset_path: str, model_names: list[str]) -> None:
    try:
        run_stt_benchmark(experiment_id, dataset_path, model_names)
    except DatasetError as exc:
        logger.error("benchmark run '%s' failed: %s", experiment_id, exc)


@router.post("", response_model=RunAccepted, status_code=202)
def start_run(request: RunRequest, background_tasks: BackgroundTasks) -> RunAccepted:
    """Kick off a benchmark run in the background. Returns immediately with
    the experiment_id -- poll GET /experiments/{experiment_id} to watch it
    move from 'running' to 'success'/'partial'/'failed'."""
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
        _run_in_background, experiment_id, request.dataset_path, request.models
    )
    return RunAccepted(experiment_id=experiment_id)


@router.get("", response_model=list[Experiment])
def list_experiments_endpoint() -> list[Experiment]:
    init_db()
    with session_scope() as session:
        return list_experiments(session)


@router.get("/{experiment_id}", response_model=ExperimentDetail)
def get_experiment_endpoint(experiment_id: str) -> ExperimentDetail:
    init_db()
    with session_scope() as session:
        experiment = get_experiment(session, experiment_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
        results = get_results_for_experiment(session, experiment_id)
    return ExperimentDetail(experiment=experiment, results=results)


@router.get("/{experiment_id}/report", response_class=PlainTextResponse)
def get_experiment_report(experiment_id: str) -> str:
    """Markdown report for one experiment -- same aggregations as the
    dashboard's summary view, rendered as plain-text tables."""
    init_db()
    with session_scope() as session:
        experiment = get_experiment(session, experiment_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
        results = get_results_for_experiment(session, experiment_id)
    return generate_markdown_report(experiment, results)
