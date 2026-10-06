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

    robustness = report["cellwise_loss_robustness"]
    if robustness["nonzero_cells"] <= 0:
        raise AssertionError("No non-zero loss cells were audited.")
    if robustness["perturbations_per_case"] != 2 * robustness["nonzero_cells"]:
        raise AssertionError(robustness)
    if robustness["decisions_tested"] != (
        report["n_cases"] * robustness["perturbations_per_case"]
    ):
        raise AssertionError(robustness)
    if not (0.0 <= robustness["decision_stability_rate"] <= 1.0):
        raise AssertionError(robustness["decision_stability_rate"])
    if set(robustness["by_stratum"]) != {"empathetic", "task_general"}:
        raise AssertionError(robustness["by_stratum"])

    grouped = report["grouped_cost_sensitivity"]
    if grouped["factors"] != [0.8, 1.2]:
        raise AssertionError(grouped["factors"])
    expected_scenarios = {
        "clarification_cost",
        "action_vs_statement_mismatch",
        "status_vs_question_mismatch",
    }
    if set(grouped["scenarios"]) != expected_scenarios:
        raise AssertionError(grouped["scenarios"])
    for scenario in expected_scenarios:
        factors = grouped["scenarios"][scenario]
        if set(factors) != {"0.8", "1.2"}:
            raise AssertionError(factors)
        for result in factors.values():
            if not result["cells"]:
                raise AssertionError(result)
            if not (0 <= result["flips_vs_production"] <= report["n_cases"]):
                raise AssertionError(result)
            if not (0.0 <= result["flip_rate"] <= 1.0):
                raise AssertionError(result)

    print("intent risk sensitivity audit: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
