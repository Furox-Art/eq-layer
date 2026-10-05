"""Run the policy selector over every seeded case and score the chosen register.

Usage: python eval/run.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.affect import HeuristicAffect, load_cases  # noqa: E402
from eq_layer.policies import Selector  # noqa: E402
from eq_layer.steer import aggregate, score_case  # noqa: E402

CASES = os.path.join(os.path.dirname(__file__), "cases.jsonl")


def main() -> None:
    cases = load_cases(CASES)
    adapter = HeuristicAffect()
    selector = Selector()
    scores = []

    print(f"{'case':<12} {'expected':<24} {'selected':<24} ok")
    print("-" * 72)

    for case in cases:
        messages = case["transcript"]
        state = adapter.infer(messages, turn_index=len(messages))
        selection = selector.select(state)
        register = selection.policy.register.value

        expected = case["expected_policy"]
        ok = selection.policy.name == expected

        scores.append(
            score_case(
                response=selection.policy.example_opener,
                case=case,
                selected_registers=[register],
            )
        )
        print(f"{case['id']:<12} {expected:<24} {selection.policy.name:<24} {'y' if ok else 'n'}")

    print()
    print(json.dumps(aggregate(scores), indent=2))


if __name__ == "__main__":
    main()