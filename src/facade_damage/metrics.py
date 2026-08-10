"""Small metric helpers used in tests and result verification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(
    *, true_positive: int, false_positive: int, false_negative: int
) -> BinaryMetrics:
    """Compute precision, recall, and F1 from non-negative counts."""

    values = (true_positive, false_positive, false_negative)
    if any(value < 0 for value in values):
        raise ValueError("Confusion counts must be non-negative")

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return BinaryMetrics(precision=precision, recall=recall, f1=f1)
