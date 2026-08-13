"""SQLAlchemy ORM models.

Mirrors the pydantic domain schemas in core/models.py, but as persisted
tables. Kept deliberately separate from the pydantic models -- the DB
schema is allowed to diverge from the API/domain schema over time (e.g.
adding indexes, denormalizing for query speed) without touching the
domain layer.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ExperimentORM(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(String, primary_key=True)
    benchmark_name: Mapped[str] = mapped_column(String)
    benchmark_spec_version: Mapped[str] = mapped_column(String)
    dataset_version: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="running")
    start_time: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    end_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Small nested fields stored as JSON text rather than separate tables --
    # simplest thing that works for the MVP. Revisit if querying inside
    # these fields (e.g. "all experiments run under noise_10db") becomes
    # a real need.
    conditions_json: Mapped[str] = mapped_column(Text, default="[]")
    model_versions_json: Mapped[str] = mapped_column(Text, default="{}")
    hardware_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    stt_results: Mapped[list["STTResultORM"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class STTResultORM(Base):
    __tablename__ = "stt_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.experiment_id"))
    test_case_id: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    prediction: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference: Mapped[str] = mapped_column(Text)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    rtf: Mapped[float | None] = mapped_column(Float, nullable=True)
    wer: Mapped[float | None] = mapped_column(Float, nullable=True)
    cer: Mapped[float | None] = mapped_column(Float, nullable=True)
    condition: Mapped[str] = mapped_column(String, default="clean")
    failed: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    experiment: Mapped["ExperimentORM"] = relationship(back_populates="stt_results")
