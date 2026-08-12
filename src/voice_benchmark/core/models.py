"""Core domain schemas.

These are the shapes that flow through the whole system: a BenchmarkSpec
describes *what* to run, an Experiment records *that a run happened* (with
immutable context), TestCase is *one input item*, and STTResult/TTSResult
are *what came out*. Everything else in the codebase (orchestration,
benchmarks, storage) is built around these.

Reference: spec sections 8 (Benchmark Specification), 9 (Dataset/Test Case
Design), 10 (Model Registry), 15 (Experiment/Reproducibility Design).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

class BenchmarkType(str, Enum):
    STT = "stt"
    TTS = "tts"


class Condition(str, Enum):
    """Audio/noise condition a test case or benchmark run was evaluated under."""

    CLEAN = "clean"
    NOISE_20DB = "noise_20db"
    NOISE_10DB = "noise_10db"
    NOISE_5DB = "noise_5db"
    NOISE_0DB = "noise_0db"
    NOISE_NEG5DB = "noise_neg5db"


class ExperimentStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"   # some model/test-case runs failed -- never hide this
    FAILED = "failed"


# --------------------------------------------------------------------------
# Model registry / adapter metadata
# --------------------------------------------------------------------------

class ModelMetadata(BaseModel):
    """Identity + version info for one model, recorded with every result
    so a score can always be traced back to exactly what produced it."""

    name: str
    version: str | None = None
    adapter: str
    config: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Dataset / test cases
# --------------------------------------------------------------------------

class DatasetRef(BaseModel):
    name: str
    version: str
    language: str = "en"


class TestCase(BaseModel):
    """One test item. audio_path is required for STT (input audio) and
    optional for TTS (reference_text is the prompt to synthesize)."""

    id: str
    audio_path: str | None = None
    reference_text: str
    language: str = "en"
    category: str = "general"
    condition: Condition = Condition.CLEAN


# --------------------------------------------------------------------------
# Benchmark specification
# --------------------------------------------------------------------------

class BenchmarkSpec(BaseModel):
    """Versioned, declarative description of a benchmark run.

    Treated as an input to the experiment engine, not ad-hoc dashboard
    state -- the same spec should be re-runnable and produce comparable
    results.
    """

    name: str
    type: BenchmarkType
    version: str = "1.0"

    dataset: DatasetRef
    models: list[str]
    conditions: list[Condition] = Field(default_factory=lambda: [Condition.CLEAN])
    metrics: list[str]

    record_hardware: bool = True


# --------------------------------------------------------------------------
# Experiment (execution record)
# --------------------------------------------------------------------------

class HardwareMetadata(BaseModel):
    """Recorded once per experiment (section 15)."""

    os: str
    python_version: str
    cpu: str | None = None
    gpu: str | None = None
    ram_gb: float | None = None
    vram_gb: float | None = None


class Experiment(BaseModel):
    """Immutable record of one benchmark execution."""

    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    benchmark_name: str
    benchmark_spec_version: str
    dataset_version: str
    model_versions: dict[str, str | None] = Field(default_factory=dict)
    hardware: HardwareMetadata | None = None
    conditions: list[Condition]
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    status: ExperimentStatus = ExperimentStatus.RUNNING


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

class STTResult(BaseModel):
    """One STT prediction + its metrics (section 11)."""

    experiment_id: str
    test_case_id: str
    model: str
    prediction: str | None = None
    reference: str
    latency_ms: float | None = None
    audio_duration_sec: float | None = None
    rtf: float | None = None
    wer: float | None = None
    cer: float | None = None
    condition: Condition = Condition.CLEAN
    failed: bool = False
    error_message: str | None = None


class TTSResult(BaseModel):
    """One TTS generation + its objective metrics (section 12).

    Naturalness/speaker-similarity human ratings are NOT here by design --
    the spec treats those as a separate evaluation track, not folded into
    an arbitrary objective quality number.
    """

    experiment_id: str
    test_case_id: str
    model: str
    output_path: str | None = None
    generation_latency_ms: float | None = None
    output_duration_sec: float | None = None
    rtf: float | None = None
    speech_rate_wpm: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    silence_ratio: float | None = None
    failed: bool = False
    error_message: str | None = None
