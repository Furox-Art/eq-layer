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
LOSS_MATRIX_ID = "hand-specified-v1"

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



def fuse_dialogue_evidence(
    belief: IntentBelief,
    *,
    text: str,
    dialogue_act: str | None = None,
    act_confidence: float = 0.0,
) -> tuple[IntentBelief, tuple[str, ...]]:
    """Fuse independent dialogue-form evidence into an intent posterior.

    This is likelihood reweighting, not a hard label override. The raw learned
    posterior should be retained separately by callers for provenance.
    """
    values = belief.as_dict()
    weights = {label: 1.0 for label in INTENT_LABELS}
    evidence: list[str] = []

    confidence = max(0.0, min(1.0, float(act_confidence)))
    if dialogue_act == "inform":
        weights["statement"] *= 1.0 + 5.0 * confidence
        evidence.append(f"dialogue_act:inform:{confidence:.3f}")
    elif dialogue_act == "question":
        weights["question"] *= 1.0 + 4.0 * confidence
        weights["status_check"] *= 1.0 + 1.0 * confidence
        weights["explanation"] *= 1.0 + 1.0 * confidence
        evidence.append(f"dialogue_act:question:{confidence:.3f}")
    elif dialogue_act == "directive":
        weights["action_request"] *= 1.0 + 5.0 * confidence
        evidence.append(f"dialogue_act:directive:{confidence:.3f}")
    elif dialogue_act == "commissive":
        weights["statement"] *= 1.0 + 2.0 * confidence
        evidence.append(f"dialogue_act:commissive:{confidence:.3f}")

    # Surface punctuation is an independent structural observation. It guards
    # against a dialogue-act classifier incorrectly calling an explicit
    # question an inform/statement.
    if text.rstrip().endswith("?"):
        weights["question"] *= 4.0
        weights["status_check"] *= 1.5
        weights["explanation"] *= 1.5
        weights["statement"] *= 0.25
        evidence.append("structure:question_mark")

    fused = IntentBelief.from_mapping(
        {
            label: values[label] * weights[label]
            for label in INTENT_LABELS
        }
    )
    return fused, tuple(evidence)

def expected_loss(
    belief: IntentBelief,
    decision: str,
    *,
    loss_matrix: dict[str, dict[str, float]] | None = None,
) -> float:
    matrix = loss_matrix or LOSS_MATRIX
    if decision not in matrix:
        raise ValueError(f"Unsupported intent decision: {decision}")
    probabilities = belief.as_dict()
    return sum(
        probabilities[label] * matrix[decision][label]
        for label in INTENT_LABELS
    )


def decide_intent_action(
    belief: IntentBelief,
    *,
    loss_matrix: dict[str, dict[str, float]] | None = None,
) -> IntentRiskDecision:
    """Choose the response move with minimum expected interaction loss.

    This is POMDP-inspired uncertainty handling, not a full POMDP: there is no
    transition model or long-horizon value function. The belief distribution is
    preserved and a transparent one-step Bayes-risk decision is made from it.
    """

    matrix = loss_matrix or LOSS_MATRIX
    losses = {
        decision: expected_loss(belief, decision, loss_matrix=matrix)
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
            f"bayes_risk:{chosen}; matrix={LOSS_MATRIX_ID}; top={belief.top_kind}:"
            f"{belief.top_probability:.4f}; entropy={belief.normalized_entropy:.4f}"
        ),
    )
