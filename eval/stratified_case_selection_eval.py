"""Regression test for balanced pilot case selection.

Usage: python eval/stratified_case_selection_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.response_experiment import (  # noqa: E402
    select_stratified_cases,
    stratum_counts,
)


def make_cases(stratum: str, n: int, offset: int) -> list[dict]:
    return [
        {
            "id": f"{stratum}-{offset + index:03d}",
            "transcript": [{"role": "user", "content": f"{stratum} {index}"}],
            "source": {"stratum": stratum},
        }
        for index in range(n)
    ]


def main() -> int:
    cases = (
        make_cases("empathetic", 60, 0)
        + make_cases("task_repair", 30, 100)
        + make_cases("task_general", 30, 200)
    )

    selected = select_stratified_cases(cases, 12)
    counts = stratum_counts(selected)
    expected = {
        "empathetic": 4,
        "task_general": 4,
        "task_repair": 4,
    }
    if counts != expected:
        raise AssertionError(f"Expected balanced 4/4/4 pilot, got {counts}")

    first_three = [
        case["source"]["stratum"]
        for case in selected[:3]
    ]
    if len(set(first_three)) != 3:
        raise AssertionError(
            f"Pilot is not round-robin interleaved across strata: {first_three}"
        )

    small = (
        make_cases("a", 1, 0)
        + make_cases("b", 5, 10)
        + make_cases("c", 5, 20)
    )
    redistributed = select_stratified_cases(small, 6)
    if len(redistributed) != 6:
        raise AssertionError("Unused quota was not redistributed.")

    try:
        select_stratified_cases(
            [{"id": "missing", "transcript": [{"role": "user", "content": "x"}]}],
            1,
        )
    except ValueError as exc:
        if "source.stratum" not in str(exc):
            raise
    else:
        raise AssertionError("Missing stratum must fail stratified selection.")

    print("stratified pilot selection: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
