"""Domain exceptions.

Per the spec's logging/error-handling rules (section 25): a benchmark run
must never silently swallow a model failure or treat it as a zero-error
score. These exceptions carry enough context (model, test case) to record
that a failure happened, not just that "something went wrong".
"""
from __future__ import annotations


class VoiceBenchmarkError(Exception):
    """Base class for all Voice Benchmark Lab errors."""


class ModelNotFoundError(VoiceBenchmarkError):
    """Raised when a requested model is not present in the registry."""


class ModelLoadError(VoiceBenchmarkError):
    """Raised when a model adapter fails to load/initialize."""


class InferenceError(VoiceBenchmarkError):
    """Raised when a model fails during transcription/synthesis.

    Always carries which model and (if applicable) which test case failed,
    so the orchestrator can record it and mark the experiment partial
    instead of dropping the failure silently.
    """

    def __init__(self, message: str, *, model: str, test_case_id: str | None = None) -> None:
        super().__init__(message)
        self.model = model
        self.test_case_id = test_case_id


class DatasetError(VoiceBenchmarkError):
    """Raised for dataset loading or manifest validation problems."""


class BenchmarkSpecError(VoiceBenchmarkError):
    """Raised when a benchmark specification fails validation."""
