"""Repository for STTResult persistence."""
from __future__ import annotations

from sqlalchemy.orm import Session

from voice_benchmark.core.models import Condition, STTResult
from voice_benchmark.storage.models import STTResultORM


def save_stt_result(session: Session, result: STTResult) -> None:
    orm = STTResultORM(
        experiment_id=result.experiment_id,
        test_case_id=result.test_case_id,
        model=result.model,
        prediction=result.prediction,
        reference=result.reference,
        latency_ms=result.latency_ms,
        audio_duration_sec=result.audio_duration_sec,
        rtf=result.rtf,
        wer=result.wer,
        cer=result.cer,
        condition=result.condition.value,
        failed=result.failed,
        error_message=result.error_message,
    )
    session.add(orm)


def _to_domain(orm: STTResultORM) -> STTResult:
    return STTResult(
        experiment_id=orm.experiment_id,
        test_case_id=orm.test_case_id,
        model=orm.model,
        prediction=orm.prediction,
        reference=orm.reference,
        latency_ms=orm.latency_ms,
        audio_duration_sec=orm.audio_duration_sec,
        rtf=orm.rtf,
        wer=orm.wer,
        cer=orm.cer,
        condition=Condition(orm.condition),
        failed=orm.failed,
        error_message=orm.error_message,
    )


def get_results_for_experiment(session: Session, experiment_id: str) -> list[STTResult]:
    orms = session.query(STTResultORM).filter_by(experiment_id=experiment_id).all()
    return [_to_domain(o) for o in orms]
