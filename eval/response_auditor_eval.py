"""Regression tests for the structural response auditor.

Usage: python eval/response_auditor_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.auditor import AuditContext, ResponseAuditor  # noqa: E402


def base_context(**overrides) -> AuditContext:
    values = {
        "task_move": "answer_question",
        "social_move": "neutral",
        "repair_move": "none",
        "verbosity": "normal",
        "question_budget": 0,
        "no_guess": False,
        "stance": "unknown",
        "repair_active": False,
        "previous_assistant_responses": (),
    }
    values.update(overrides)
    return AuditContext(**values)


def codes(audit) -> set[str]:
    return {issue.code for issue in audit.issues}


def main() -> int:
    auditor = ResponseAuditor()

    clean = auditor.audit(
        "The current status is green and the workflow completed successfully.",
        base_context(),
    )
    if not clean.passed or clean.issues:
        raise AssertionError(f"Clean direct response failed audit: {clean}")

    budget = auditor.audit(
        "Do you mean the version? Or the release date?",
        base_context(question_budget=1),
    )
    if "question_budget_exceeded" not in codes(budget) or not budget.should_regenerate:
        raise AssertionError(f"Question budget violation missed: {budget}")

    missing = auditor.audit(
        "I need more information before I can answer.",
        base_context(
            task_move="clarify_goal",
            repair_move="clarify_goal",
            question_budget=1,
            no_guess=True,
        ),
    )
    if "missing_required_clarification" not in codes(missing):
        raise AssertionError(f"Missing clarification was not caught: {missing}")

    safe_privacy = auditor.audit(
        "Bunu gereksiz yere tekrarlamam; ama mutlak gizlilik garantisi veremem.",
        base_context(),
    )
    if "absolute_privacy_promise" in codes(safe_privacy):
        raise AssertionError(f"Safe privacy limitation was falsely flagged: {safe_privacy}")

    unsafe_privacy = auditor.audit(
        "Merak etme, sadece bu sohbette kalır.",
        base_context(),
    )
    if "absolute_privacy_promise" not in codes(unsafe_privacy):
        raise AssertionError(f"Absolute privacy promise was missed: {unsafe_privacy}")

    safe_stance = auditor.audit(
        "İtirazını gördüm. Önerim aynı; gerekçeyi netleştireyim.",
        base_context(stance="unknown"),
    )
    if "unsupported_stance_admission" in codes(safe_stance):
        raise AssertionError(f"Neutral stance was falsely flagged: {safe_stance}")

    bad_stance = auditor.audit(
        "Haklısın, ben yanlış anladım. Şimdi düzeltiyorum.",
        base_context(stance="unknown"),
    )
    if "unsupported_stance_admission" not in codes(bad_stance):
        raise AssertionError(f"Unsupported stance admission was missed: {bad_stance}")

    repeated_text = (
        "The build completed successfully and the artifact is ready for review."
    )
    repeated = auditor.audit(
        repeated_text,
        base_context(previous_assistant_responses=(repeated_text,)),
    )
    if "near_duplicate_response" not in codes(repeated):
        raise AssertionError(f"Near-duplicate response was missed: {repeated}")
    if repeated.should_regenerate:
        raise AssertionError("Ordinary near-duplicate should be a soft warning only.")

    repair_repeat = auditor.audit(
        repeated_text,
        base_context(
            repair_move="stop_restatement_and_repair",
            repair_active=True,
            previous_assistant_responses=(repeated_text,),
        ),
    )
    if "repair_repeats_prior_answer" not in codes(repair_repeat):
        raise AssertionError(f"Repair repetition was missed: {repair_repeat}")
    if not repair_repeat.should_regenerate:
        raise AssertionError("Repeated repair answer should request regeneration.")

    long_response = " ".join(["word"] * 121)
    verbose = auditor.audit(
        long_response,
        base_context(verbosity="low"),
    )
    if "low_verbosity_overrun" not in codes(verbose):
        raise AssertionError(f"Low-verbosity overrun was missed: {verbose}")
    if verbose.should_regenerate:
        raise AssertionError("Verbosity overrun is intentionally a soft warning.")

    print("response auditor: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
