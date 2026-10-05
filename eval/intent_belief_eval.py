"""Regression tests for intent belief-state Bayes-risk routing.

Usage: python eval/intent_belief_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.intent_belief import (  # noqa: E402
    IntentBelief,
    decide_intent_action,
)


def decision(values: dict[str, float]):
    belief = IntentBelief.from_mapping(values)
    return belief, decide_intent_action(belief)


def main() -> int:
    clear, clear_decision = decision(
        {
            "action_request": 0.85,
            "status_check": 0.03,
            "explanation": 0.03,
            "question": 0.03,
            "statement": 0.03,
            "unknown": 0.03,
        }
    )
    if clear_decision.action != "action_request":
        raise AssertionError(f"Clear action did not execute: {clear_decision}")
    if clear.top_kind != "action_request":
        raise AssertionError(f"Wrong top belief: {clear.top_kind}")

    # A simple margin threshold would be tempted to clarify here because the
    # top two intents are close. Bayes risk may answer directly because status
    # and generic-question responses have low semantic mismatch cost.
    adjacent, adjacent_decision = decision(
        {
            "action_request": 0.04,
            "status_check": 0.42,
            "explanation": 0.06,
            "question": 0.40,
            "statement": 0.04,
            "unknown": 0.04,
        }
    )
    if adjacent.margin >= 0.08:
        raise AssertionError("Adjacent ambiguity test is not actually low-margin.")
    if adjacent_decision.action != "question":
        raise AssertionError(
            f"Adjacent low-cost ambiguity should answer, got {adjacent_decision}"
        )

    # Ambiguity between acting and merely responding to a statement is costly;
    # clarification should beat guessing even though one class is marginally top.
    costly, costly_decision = decision(
        {
            "action_request": 0.45,
            "status_check": 0.04,
            "explanation": 0.03,
            "question": 0.05,
            "statement": 0.40,
            "unknown": 0.03,
        }
    )
    if costly.top_kind != "action_request":
        raise AssertionError(f"Costly ambiguity top class changed: {costly.top_kind}")
    if costly_decision.action != "clarify":
        raise AssertionError(
            f"High-cost ambiguity should clarify, got {costly_decision}"
        )

    unknown, unknown_decision = decision(
        {
            "action_request": 0.05,
            "status_check": 0.10,
            "explanation": 0.05,
            "question": 0.10,
            "statement": 0.20,
            "unknown": 0.50,
        }
    )
    if unknown_decision.action != "clarify":
        raise AssertionError(f"Unknown-heavy belief should clarify: {unknown_decision}")

    # Normalization is part of the public contract.
    normalized = IntentBelief.from_mapping(
        {"question": 2.0, "status_check": 1.0}
    )
    if abs(sum(normalized.as_dict().values()) - 1.0) > 1e-12:
        raise AssertionError("Belief probabilities do not normalize to 1.")
    if not (0.0 <= normalized.normalized_entropy <= 1.0):
        raise AssertionError("Normalized entropy is outside [0, 1].")

    print("intent belief Bayes-risk routing: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
