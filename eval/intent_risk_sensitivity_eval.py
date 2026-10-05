"""Regression tests for the intent Bayes-risk sensitivity audit.

Usage: python eval/intent_risk_sensitivity_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from intent_risk_sensitivity import audit  # noqa: E402


def main() -> int:
    pairs = [
        {
            "id": "case-a",
            "source": {"stratum": "empathetic"},
            "generation": {
                "intent_belief_raw": {
                    "action_request": 0.05,
                    "status_check": 0.05,
                    "explanation": 0.05,
                    "question": 0.05,
                    "statement": 0.30,
                    "unknown": 0.50,
                },
                "intent_belief": {
                    "action_request": 0.02,
                    "status_check": 0.03,
                    "explanation": 0.03,
                    "question": 0.02,
                    "statement": 0.80,
                    "unknown": 0.10,
                },
                "intent_belief_entropy": 0.50,
                "intent_risk_decision": {"action": "statement"},
            },
        },
        {
            "id": "case-b",
            "source": {"stratum": "task_general"},
            "generation": {
                "intent_belief_raw": {
                    "action_request": 0.05,
                    "status_check": 0.20,
                    "explanation": 0.05,
                    "question": 0.15,
                    "statement": 0.10,
                    "unknown": 0.45,
                },
                "intent_belief": {
                    "action_request": 0.03,
                    "status_check": 0.15,
                    "explanation": 0.05,
                    "question": 0.07,
                    "statement": 0.10,
                    "unknown": 0.60,
                },
                "intent_belief_entropy": 0.70,
                "intent_risk_decision": {"action": "clarify"},
            },
        },
    ]

    report = audit(
        pairs,
        multipliers=(1.0, 1.5, 2.5),
    )

    if report["n_cases"] != 2:
        raise AssertionError(report)
    if report["production_action_counts"] != {
        "statement": 1,
        "clarify": 1,
    }:
        raise AssertionError(report["production_action_counts"])
    if report["production_clarification_rate"] != 0.5:
        raise AssertionError(report["production_clarification_rate"])
    if report["raw_to_fused_action_flips"] < 1:
        raise AssertionError("Expected at least one raw->fused routing change.")
    if set(report["by_stratum"]) != {"empathetic", "task_general"}:
        raise AssertionError(report["by_stratum"])
    if "1.5" not in report["clarify_cost_sensitivity"]:
        raise AssertionError("Sensitivity multiplier missing.")

    print("intent risk sensitivity audit: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
