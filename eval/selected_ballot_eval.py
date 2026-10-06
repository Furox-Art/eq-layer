"""Check the primary-endpoint ballot is built from the control, not the baseline.

The whole point of the calibrated control is that the ballot compares EQ against
a same-length response. Building it from the raw baseline would look identical
in code and quietly reintroduce the length confound, so these tests pin which
arm ends up in the ballot and what the decode key calls it.

Usage: python eval/selected_ballot_eval.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.ab_eval import prepare_blinded  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def main() -> int:
    pairs = [
        {"id": "a", "context": [], "baseline": "RAW RAW RAW", "eq": "SHORT"},
        {"id": "b", "context": [], "baseline": "RAW TWO", "eq": "ALSO SHORT"},
    ]

    # Default naming must not change.
    _, default_key = prepare_blinded(pairs, seed=1)
    check(
        set(v for row in default_key.values() for v in row.values())
        == {"eq", "baseline"},
        "default arm names must stay eq/baseline",
    )

    # Naming the control explicitly must survive into the decode key.
    arm = "length_matched:one_sentence"
    ballot, key = prepare_blinded(pairs, seed=1, arm_names=("eq", arm))
    values = {v for row in key.values() for v in row.values()}
    check(values == {"eq", arm}, f"key should name both arms, got {values}")
    check("baseline" not in values, "the control arm must not be labelled baseline")

    # The raw baseline must never leak into the ballot.
    for row in ballot:
        check(
            "RAW" not in row["response_A"] and "RAW" not in row["response_B"],
            f"{row['id']}: raw baseline leaked into the ballot",
        )
        check(
            row["response_A"].strip() != "" and row["response_B"].strip() != "",
            f"{row['id']}: blank response",
        )

    # Side assignment must be randomized, not fixed.
    sides = set()
    for seed in range(12):
        _, k = prepare_blinded(pairs, seed=seed, arm_names=("eq", arm))
        sides.add(tuple(sorted(k["a"].items())))
    check(len(sides) > 1, f"A/B assignment looks fixed across seeds: {sides}")

    # Arm names must be exactly two or the swap breaks.
    try:
        prepare_blinded(pairs, seed=1, arm_names=("eq",))
        FAILURES.append("a single arm name should raise")
    except (ValueError, TypeError, IndexError):
        pass

    print(json.dumps({"failures": FAILURES, "ok": not FAILURES}, indent=2))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())