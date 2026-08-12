"""STT evaluation metrics: WER, CER, and text normalization.

WER = (Substitutions + Deletions + Insertions) / Number of reference words
(spec section 14). Text must be normalized *consistently* before comparing,
or differences in casing/punctuation get counted as transcription errors
they aren't. The normalization rules are intentionally explicit and
documented here rather than left implicit in a library default.
"""
from __future__ import annotations

import re
import string

import jiwer

from voice_benchmark.core.exceptions import VoiceBenchmarkError

# Transform applied to both prediction and reference before scoring:
# lowercase, strip punctuation, collapse whitespace, remove leading/trailing
# space. This is deliberately conservative -- it does NOT expand numbers,
# spell out abbreviations, etc. Document any change here, since it changes
# every historical score's comparability.
_NORMALIZE_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


class MetricComputationError(VoiceBenchmarkError):
    """Raised when WER/CER cannot be computed (e.g. empty reference)."""


def normalize_text(text: str) -> str:
    """Apply the same normalization used for scoring, for display/debugging."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, prediction: str) -> float:
    """Word Error Rate. Raises MetricComputationError on an empty reference
    (WER is undefined / infinite in that case -- never silently return 0)."""
    if not reference or not reference.strip():
        raise MetricComputationError("Cannot compute WER: empty reference text")
    prediction = prediction or ""
    return jiwer.wer(
        reference,
        prediction,
        reference_transform=_NORMALIZE_TRANSFORM,
        hypothesis_transform=_NORMALIZE_TRANSFORM,
    )


def compute_cer(reference: str, prediction: str) -> float:
    """Character Error Rate."""
    if not reference or not reference.strip():
        raise MetricComputationError("Cannot compute CER: empty reference text")
    prediction = prediction or ""
    return jiwer.cer(normalize_text(reference), normalize_text(prediction))


def compute_stt_metrics(reference: str, prediction: str) -> dict[str, float]:
    """Convenience wrapper returning both metrics together."""
    return {
        "wer": compute_wer(reference, prediction),
        "cer": compute_cer(reference, prediction),
    }
