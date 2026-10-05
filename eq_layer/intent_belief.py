from __future__ import annotations

import math
from dataclasses import dataclass


INTENT_LABELS = (
    "action_request",
    "status_check",
    "explanation",
    "question",
    "statement",
    "unknown",
)

DECISIONS = (
    "action_request",
    "status_check",
    "explanation",
    "question",
    "statement",
    "clarify",
)


# Transparent Bayes-risk loss matrix.
#
# Rows are response decisions; columns are the user's true intent. Costs are
# relative interaction losses, not learned probabilities. Adjacent semantic
# confusions (e.g. status_check vs question) cost less than executing an action
# when the user merely made a statement. Clarification has a moderate cost for
# clear requests and a very low cost when the true state is unknown.
LOSS_MATRIX: dict[str, dict[str, float]] = {
    "action_request": {
        "action_request": 0.00,
        "status_check": 0.80,
        "explanation": 0.80,
        "question": 0.75,
        "statement": 0.90,
        "unknown": 1.00,
    },
    "status_check": {
        "action_request": 0.85,
        "status_check": 0.00,
        "explanation": 0.50,
        "question": 0.35,
        "statement": 0.75,
        "unknown": 0.90,
    },
    "explanation": {
        "action_request": 0.80,
        "status_check": 0.50,
        "explanation": 0.00,
        "question": 0.30,
        "statement": 0.65,
        "unknown": 0.85,
    },
    "question": {
        "action_request": 0.75,
        "status_check": 0.35,
        "explanation": 0.30,
        "question": 0.00,
        "statement": 0.55,
        "unknown": 0.80,
    },
    "statement": {
        "action_request": 0.80,
        "status_check": 0.70,
        "explanation": 0.60,
        "question": 0.50,
        "statement": 0.00,
        "unknown": 0.70,
    },
    "clarify": {
        "action_request": 0.38,
        "status_check": 0.38,
        "explanation": 0.38,
        "question": 0.38,
        "statement": 0.48,
        "unknown": 0.05,
    },
}


@dataclass(frozen=True)
class IntentBelief:
    """Normalised posterior belief over mutually exclusive intent classes."""

    probabilities: tuple[tuple[str, float], ...]

    @classmethod
    def from_mapping(cls, values: dict[str, float]) -> "IntentBelief":
        unknown = set(values) - set(INTENT_LABELS)
        if unknown:
            raise ValueError(f"Unsupported intent labels: {sorted(unknown)}")

        raw = {label: max(0.0, float(values.get(label, 0.0))) for label in INTENT_LABELS}
        total = sum(raw.values())
        if total <= 0:
            raw = {label: (1.0 if label == "unknown" else 0.0) for label in INTENT_LABELS}
            total = 1.0

        normalized = tuple(
            (label, raw[label] / total)
            for label in INTENT_LABELS
        )
        return cls(probabilities=normalized)

    def as_dict(self) -> dict[str, float]:
        return {label: probability for label, probability in self.probabilities}

    def probability(self, label: str) -> float:
        return self.as_dict().get(label, 0.0)

    @property
    def ranked(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            sorted(
                self.probabilities,
                key=lambda item: (-item[1], item[0]),
            )
        )

    @property
    def top_kind(self) -> str:
        return self.ranked[0][0]

    @property
    def top_probability(self) -> float:
        return self.ranked[0][1]

    @property
    def runner_up_kind(self) -> str:
        return self.ranked[1][0]

    @property
    def margin(self) -> float:
        return self.ranked[0][1] - self.ranked[1][1]

    @property
    def normalized_entropy(self) -> float:
        values = [probability for _, probability in self.probabilities if probability > 0]
        if len(values) <= 1:
            return 0.0
        entropy = -sum(probability * math.log(probability) for probability in values)
        return entropy / math.log(len(INTENT_LABELS))


@dataclass(frozen=True)
class IntentRiskDecision:
    """Bayes-risk choice over response moves given an intent belief state."""

    action: str
    selected_intent: str | None
    expected_loss: float
    losses: tuple[tuple[str, float], ...]
    rationale: str

    def losses_dict(self) -> dict[str, float]:
        return dict(self.losses)


def expected_loss(belief: IntentBelief, decision: str) -> float:
    if decision not in LOSS_MATRIX:
        raise ValueError(f"Unsupported intent decision: {decision}")
    probabilities = belief.as_dict()
    return sum(
        probabilities[label] * LOSS_MATRIX[decision][label]
        for label in INTENT_LABELS
    )


def decide_intent_action(belief: IntentBelief) -> IntentRiskDecision:
    """Choose the response move with minimum expected interaction loss.

    This is POMDP-inspired uncertainty handling, not a full POMDP: there is no
    transition model or long-horizon value function. The belief distribution is
    preserved and a transparent one-step Bayes-risk decision is made from it.
    """

    losses = {
        decision: expected_loss(belief, decision)
        for decision in DECISIONS
    }
    best_loss = min(losses.values())
    winners = [
        decision
        for decision, loss in losses.items()
        if abs(loss - best_loss) <= 1e-12
    ]

    # Epistemically conservative tie-break: clarification wins an exact risk
    # tie. Otherwise choose deterministically by declaration order.
    if "clarify" in winners:
        chosen = "clarify"
    else:
        chosen = next(decision for decision in DECISIONS if decision in winners)

    selected_intent = None if chosen == "clarify" else chosen
    return IntentRiskDecision(
        action=chosen,
        selected_intent=selected_intent,
        expected_loss=round(losses[chosen], 6),
        losses=tuple(
            (decision, round(losses[decision], 6))
            for decision in DECISIONS
        ),
        rationale=(
            f"bayes_risk:{chosen}; top={belief.top_kind}:"
            f"{belief.top_probability:.4f}; entropy={belief.normalized_entropy:.4f}"
        ),
    )
