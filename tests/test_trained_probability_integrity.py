"""Learned binary adapters must not turn invalid model scores into decisions."""

from __future__ import annotations

import math

import pytest

from eq_layer.trained_breakdown import TrainedBreakdown
from eq_layer.trained_disagreement import TrainedDisagreement


class FixedProbabilityEstimator:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, _texts: list[str]) -> list[list[float]]:
        return [[1.0 - self.probability, self.probability]]


@pytest.mark.parametrize("adapter", [TrainedBreakdown, TrainedDisagreement])
@pytest.mark.parametrize("probability", [math.nan, math.inf, -math.inf, -0.01, 1.01])
def test_unusable_model_probability_never_emits_a_decision(
    adapter: type, probability: float
) -> None:
    model = adapter(model=FixedProbabilityEstimator(probability))
    with pytest.raises(ValueError, match="probability"):
        model.predict("user text")


@pytest.mark.parametrize("adapter", [TrainedBreakdown, TrainedDisagreement])
@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf, -0.01, 1.01])
def test_invalid_decision_threshold_never_emits_a_decision(adapter: type, threshold: float) -> None:
    model = adapter(model=FixedProbabilityEstimator(0.7), threshold=threshold)
    with pytest.raises(ValueError, match="threshold"):
        model.predict("user text")


@pytest.mark.parametrize("adapter", [TrainedBreakdown, TrainedDisagreement])
def test_exact_probability_boundaries_are_accepted(adapter: type) -> None:
    for probability in (0.0, 1.0):
        model = adapter(model=FixedProbabilityEstimator(probability))
        decision = model.predict("user text")
        assert decision.probability == probability
        label = decision.is_breakdown if adapter is TrainedBreakdown else decision.is_disagreement
        assert label is (probability >= model.threshold)
