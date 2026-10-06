"""Regression tests for the independent loss-matrix calibration estimator."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.intent_belief import DECISIONS, INTENT_LABELS  # noqa: E402
from eval.loss_matrix_calibration import estimate_candidate  # noqa: E402


def _synthetic_rows() -> list[dict]:
    rows: list[dict] = []
    for intent in INTENT_LABELS:
        case_id = f"cal-{intent}"
        ideal = "clarify" if intent == "unknown" else intent
        for rater in ("r1", "r2", "r3"):
            costs = {
                decision: (0 if decision == ideal else 2)
                for decision in DECISIONS
            }
            rows.append(
                {
                    "case_id": case_id,
                    "source_id": f"source-{intent}",
                    "rater_id": rater,
                    "true_intent": intent,
                    "decision_costs": costs,
                }
            )
    return rows


def main() -> int:
    rows = _synthetic_rows()
    report = estimate_candidate(
        rows,
        min_raters_per_case=3,
        min_cases_per_intent=1,
    )

    if report["status"] != "candidate-only-not-production":
        raise AssertionError(report["status"])
    if report["n_cases"] != len(INTENT_LABELS):
        raise AssertionError(report["n_cases"])
    if report["n_rating_rows"] != len(INTENT_LABELS) * 3:
        raise AssertionError(report["n_rating_rows"])

    matrix = report["loss_matrix"]
    for intent in INTENT_LABELS:
        ideal = "clarify" if intent == "unknown" else intent
        if matrix[ideal][intent] != 0.0:
            raise AssertionError(
                f"ideal synthetic cell is not zero: {ideal}/{intent}"
            )

    try:
        estimate_candidate(
            rows,
            forbidden_case_ids={"cal-action_request"},
            min_raters_per_case=3,
            min_cases_per_intent=1,
        )
    except ValueError as exc:
        if "forbidden A/B case overlap" not in str(exc):
            raise
    else:
        raise AssertionError("Forbidden case overlap was not rejected")

    duplicate = [*rows, dict(rows[0])]
    try:
        estimate_candidate(
            duplicate,
            min_raters_per_case=3,
            min_cases_per_intent=1,
        )
    except ValueError as exc:
        if "duplicate case/rater pair" not in str(exc):
            raise
    else:
        raise AssertionError("Duplicate rater/case pair was not rejected")

    print("loss matrix calibration scaffold: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
