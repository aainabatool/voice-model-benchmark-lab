"""Repository for Experiment persistence.

Translates between the pydantic Experiment domain model and the
ExperimentORM table row. Callers work with Experiment objects only --
nothing outside this file touches ExperimentORM directly.
"""
from __future__ import annotations

import json

from sqlalchemy.orm import Session

from voice_benchmark.core.models import Experiment, ExperimentStatus, HardwareMetadata
from voice_benchmark.storage.models import ExperimentORM


def save_experiment(session: Session, experiment: Experiment) -> None:
    """Insert or update (upsert-by-primary-key) an experiment record."""
    orm = session.get(ExperimentORM, experiment.experiment_id)
    if orm is None:
        orm = ExperimentORM(experiment_id=experiment.experiment_id)
        session.add(orm)

    orm.benchmark_name = experiment.benchmark_name
    orm.benchmark_spec_version = experiment.benchmark_spec_version
    orm.dataset_version = experiment.dataset_version
    orm.status = experiment.status.value
    orm.start_time = experiment.start_time
    orm.end_time = experiment.end_time
    orm.conditions_json = json.dumps([c.value for c in experiment.conditions])
    orm.model_versions_json = json.dumps(experiment.model_versions)
    orm.hardware_json = experiment.hardware.model_dump_json() if experiment.hardware else None


def _to_domain(orm: ExperimentORM) -> Experiment:
    return Experiment(
        experiment_id=orm.experiment_id,
        benchmark_name=orm.benchmark_name,
        benchmark_spec_version=orm.benchmark_spec_version,
        dataset_version=orm.dataset_version,
        model_versions=json.loads(orm.model_versions_json),
        hardware=(
            HardwareMetadata.model_validate_json(orm.hardware_json) if orm.hardware_json else None
        ),
        conditions=json.loads(orm.conditions_json),
        start_time=orm.start_time,
        end_time=orm.end_time,
        status=ExperimentStatus(orm.status),
    )


def get_experiment(session: Session, experiment_id: str) -> Experiment | None:
    orm = session.get(ExperimentORM, experiment_id)
    return _to_domain(orm) if orm is not None else None


def list_experiments(session: Session) -> list[Experiment]:
    orms = session.query(ExperimentORM).order_by(ExperimentORM.start_time.desc()).all()
    return [_to_domain(o) for o in orms]
