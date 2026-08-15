"""Pareto frontier analysis for accuracy vs. speed tradeoffs.

Given per-model points on two "lower is better" axes -- e.g. (avg_wer,
avg_rtf) -- identifies the Pareto-optimal set: models where no other
model is at least as good on both axes and strictly better on at least
one. This is what actually answers "which models are worth considering"
rather than a single ranked list, since speed and accuracy are a genuine
tradeoff, not one true winner -- a model can lose on raw accuracy and
still be Pareto-optimal because nothing beats it on the speed/accuracy
combination it offers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoPoint:
    name: str
    x: float  # e.g. avg_wer -- lower is better
    y: float  # e.g. avg_rtf -- lower is better


def compute_pareto_frontier(points: list[ParetoPoint]) -> list[str]:
    """Return the names of Pareto-optimal points.

    A point is dominated (excluded) if some other point has x<=x AND
    y<=y, with at least one strictly less -- i.e. another model matches
    or beats it on both axes and actually wins on at least one.
    """
    optimal: list[str] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q.x <= p.x and q.y <= p.y and (q.x < p.x or q.y < p.y):
                dominated = True
                break
        if not dominated:
            optimal.append(p.name)
    return optimal


def rank_by_weighted_score(
    points: list[ParetoPoint], accuracy_weight: float = 0.5
) -> list[tuple[str, float]]:
    """Normalize both axes to [0, 1] and combine into a single score
    (lower is better), sorted best-first.

    This is a secondary, simpler view alongside the Pareto set -- useful
    when someone wants "just tell me the best one" with an explicit
    accuracy/speed weighting, rather than a frontier to reason about
    themselves. accuracy_weight=0.5 weighs both equally; closer to 1.0
    prioritizes accuracy, closer to 0.0 prioritizes speed.
    """
    if not points:
        return []

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    def norm(v: float, lo: float, hi: float) -> float:
        return 0.0 if hi == lo else (v - lo) / (hi - lo)

    scored = [
        (
            p.name,
            accuracy_weight * norm(p.x, x_min, x_max)
            + (1.0 - accuracy_weight) * norm(p.y, y_min, y_max),
        )
        for p in points
    ]
    return sorted(scored, key=lambda t: t[1])
