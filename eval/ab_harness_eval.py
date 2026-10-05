"""Smoke-test the blind A/B evaluation harness.

This is engineering validation only, not a scientific result.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.ab_eval import (  # noqa: E402
    DIMENSIONS,
    prepare_blinded,
    score_blinded,
    score_multiple_blinded,
)


def main() -> int:
    pairs = [
        {
            "id": "case-1",
            "context": [{"role": "user", "content": "Status?"}],
            "baseline": "I understand this is frustrating.",
            "eq": "The run is still active.",
        },
        {
            "id": "case-2",
            "context": [{"role": "user", "content": "Explain why."}],
            "baseline": "That sounds difficult.",
            "eq": "It failed because the input schema changed.",
        },
    ]
    ballot, key = prepare_blinded(pairs, seed=7)

    if len(ballot) != 2 or set(key) != {"case-1", "case-2"}:
        raise AssertionError("A/B preparation failed.")

    # Simulate a rater choosing the EQ response on every dimension.
    for row in ballot:
        eq_side = "A" if key[row["id"]]["A"] == "eq" else "B"
        row["ratings"] = {dimension: eq_side for dimension in DIMENSIONS}

    report = score_blinded(ballot, key)
    overall = report["dimensions"]["overall"]

    if overall["eq_wins"] != 2 or overall["eq_losses"] != 0:
        raise AssertionError(f"Unexpected decoded preference: {overall}")
    if overall["eq_win_rate_non_tie"] != 1.0:
        raise AssertionError(f"Unexpected win rate: {overall}")

    # Multi-rater smoke: perfect agreement across more than one category.
    multi_first = []
    multi_second = []
    for row in ballot:
        case_id = row["id"]
        eq_side = "A" if key[case_id]["A"] == "eq" else "B"
        baseline_side = "B" if eq_side == "A" else "A"
        chosen = eq_side if case_id == "case-1" else baseline_side
        ratings = {dimension: chosen for dimension in DIMENSIONS}
        multi_first.append({**row, "ratings": dict(ratings)})
        multi_second.append({**row, "ratings": dict(ratings)})

    multi = score_multiple_blinded(
        {"rater-1": multi_first, "rater-2": multi_second},
        key,
    )
    multi_overall = multi["dimensions"]["overall"]
    if (
        multi_overall["case_majority_eq_wins"] != 1
        or multi_overall["case_majority_eq_losses"] != 1
    ):
        raise AssertionError(f"Unexpected multi-rater majority: {multi_overall}")
    if multi_overall["fleiss_kappa"] != 1.0:
        raise AssertionError(f"Unexpected Fleiss kappa: {multi_overall}")

    print("blind A/B harness: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
