"""Run the policy selector over every seeded case and report the selection table.

Usage: python eval/run.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.affect import HeuristicAffect, load_cases  # noqa: E402
from eq_layer.trained_intent import TrainedIntent  # noqa: E402
from eq_layer.policies import POLICIES, Selector  # noqa: E402
from eq_layer.steer import score_case  # noqa: E402

CASES = os.path.join(os.path.dirname(__file__), "cases.jsonl")


def main() -> int:
    cases = load_cases(CASES)
    adapter = HeuristicAffect()
    intent_adapter = TrainedIntent.from_bundled()
    selector = Selector()

    scores = []
    selected_names = []
    mismatches = []

    print(f"{'case':<10} {'subtext':<19} {'expected':<26} {'selected':<26} ok")
    print("-" * 96)

    for case in cases:
        messages = case["transcript"]
        state = adapter.infer(messages, turn_index=len(messages), annotated=case.get("annotated"))
        intent = intent_adapter.infer(messages)
        selection = selector.select(state, intent)
        selected_names.append(selection.policy.name)

        expected = case["expected_policy"]
        expected_intent = case.get("expected_intent")
        policy_ok = selection.policy.name == expected
        intent_ok = expected_intent is None or intent.kind == expected_intent
        ok = policy_ok and intent_ok
        if not ok:
            mismatches.append((case["id"], expected, selection.policy.name, expected_intent, intent.kind))

        scores.append(
            score_case(
                response=selection.policy.example_opener,
                case=case,
                selected_registers=[selection.policy.register.value],
            )
        )
        print(
            f"{case['id']:<10} {state.subtext:<19} {expected:<26} "
            f"{selection.policy.name:<26} {'y' if ok else 'n'}"
        )

    unselected = [p.name for p in POLICIES if p.name not in selected_names]

    print()
    print(f"policy selection: {len(cases) - len(mismatches)}/{len(cases)}")
    print(f"unreachable policies: {', '.join(unselected) if unselected else 'none'}")

    print()
    print(json.dumps(
        {
            "n": len(scores),
            "specificity": round(sum(s["specificity"] for s in scores) / len(scores), 3),
            "genericness": round(sum(s["genericness"] for s in scores) / len(scores), 3),
        },
        indent=2,
    ))

    if mismatches:
        print()
        print("mismatches:")
        for case_id, expected, got, expected_intent, got_intent in mismatches:
            extra = "" if expected_intent is None else f"; intent {expected_intent} -> {got_intent}"
            print(f"  {case_id}: expected {expected}, got {got}{extra}")

    return 1 if mismatches or unselected else 0


if __name__ == "__main__":
    raise SystemExit(main())