import pytest

from facade_damage.metrics import binary_metrics


def test_binary_metrics() -> None:
    result = binary_metrics(true_positive=80, false_positive=20, false_negative=10)
    assert result.precision == pytest.approx(0.8)
    assert result.recall == pytest.approx(80 / 90)
    assert result.f1 == pytest.approx(2 * 0.8 * (80 / 90) / (0.8 + 80 / 90))


def test_binary_metrics_zero_denominators() -> None:
    result = binary_metrics(true_positive=0, false_positive=0, false_negative=0)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_binary_metrics_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        binary_metrics(true_positive=1, false_positive=-1, false_negative=0)

