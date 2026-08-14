"""Repository for TTSResult persistence."""
from __future__ import annotations

from sqlalchemy.orm import Session

from voice_benchmark.core.models import TTSResult
from voice_benchmark.storage.models import TTSResultORM


def save_tts_result(session: Session, result: TTSResult) -> None:
    orm = TTSResultORM(
        experiment_id=result.experiment_id,
        test_case_id=result.test_case_id,
        model=result.model,
        output_path=result.output_path,
        generation_latency_ms=result.generation_latency_ms,
        output_duration_sec=result.output_duration_sec,
        rtf=result.rtf,
        speech_rate_wpm=result.speech_rate_wpm,
        sample_rate=result.sample_rate,
        channels=result.channels,
        silence_ratio=result.silence_ratio,
        failed=result.failed,
        error_message=result.error_message,
    )
    session.add(orm)


def _to_domain(orm: TTSResultORM) -> TTSResult:
    return TTSResult(
        experiment_id=orm.experiment_id,
        test_case_id=orm.test_case_id,
        model=orm.model,
        output_path=orm.output_path,
        generation_latency_ms=orm.generation_latency_ms,
        output_duration_sec=orm.output_duration_sec,
        rtf=orm.rtf,
        speech_rate_wpm=orm.speech_rate_wpm,
        sample_rate=orm.sample_rate,
        channels=orm.channels,
        silence_ratio=orm.silence_ratio,
        failed=orm.failed,
        error_message=orm.error_message,
    )


def get_tts_results_for_experiment(session: Session, experiment_id: str) -> list[TTSResult]:
    orms = session.query(TTSResultORM).filter_by(experiment_id=experiment_id).all()
    return [_to_domain(o) for o in orms]
