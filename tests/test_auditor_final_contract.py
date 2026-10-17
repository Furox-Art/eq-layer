from dataclasses import replace

from eq_layer.auditor import (
    AuditContext,
    ResponseAuditor,
    _max_repeat_similarity,
    audit_from_generation_metadata,
    regeneration_instruction,
)


def context(**kwargs):
    return replace(
        AuditContext(
            task_move="execute_request",
            social_move="neutral",
            repair_move="none",
            verbosity="normal",
            question_budget=1,
            no_guess=False,
        ),
        **kwargs,
    )


def codes(result):
    return {issue.code for issue in result.issues}


def test_empty_and_unbounded_questions_are_hard_failures():
    result = ResponseAuditor().audit("", context())
    assert "empty_response" in codes(result)
    assert result.should_regenerate and not result.passed
    result = ResponseAuditor().audit("Why? How?", context(question_budget=1))
    assert "question_budget_exceeded" in codes(result)


def test_required_clarification_cannot_be_skipped():
    for attr in [
        {"task_move": "clarify_goal"},
        {"repair_move": "clarify_one"},
        {"task_move": "clarify_correction"},
    ]:
        result = ResponseAuditor().audit("No question here.", context(**attr))
        assert "missing_required_clarification" in codes(result)


def test_privacy_and_stance_admissions_require_external_evidence():
    result = ResponseAuditor().audit(
        "I guarantee confidentiality. You're right.",
        context(question_budget=0),
    )
    assert "absolute_privacy_promise" in codes(result)
    assert "unsupported_stance_admission" in codes(result)
    assert "unsupported_stance_admission" not in codes(
        ResponseAuditor().audit("You're right.", context(stance="user_right"))
    )


def test_low_verbosity_and_high_similarity_are_reported():
    long = " ".join(["This is a detailed sentence"] * 35)
    result = ResponseAuditor().audit(
        long, context(verbosity="low", previous_assistant_responses=(long,))
    )
    assert "low_verbosity_overrun" in codes(result)
    assert "near_duplicate_response" in codes(result)
    assert result.soft_warnings >= 2


def test_repair_that_repeats_prior_answer_is_a_hard_failure():
    message = "The same previous answer is repeated without correction."
    result = ResponseAuditor().audit(
        message,
        context(
            repair_active=True,
            repair_move="repair_targeted",
            previous_assistant_responses=(message,),
        ),
    )
    assert "repair_repeats_prior_answer" in codes(result)


def test_similarity_skips_blank_previous_messages():
    assert _max_repeat_similarity("", ("something",)) == 0.0
    assert _max_repeat_similarity("response", ()) == 0.0
    assert _max_repeat_similarity("response", ("  ",)) == 0.0


def test_regeneration_uses_hard_errors_only_and_never_invents_new_claims():
    empty = ResponseAuditor().audit("Hello", context())
    assert regeneration_instruction(empty) == ""
    failed = ResponseAuditor().audit(
        "No one else will know. What? Why?", context(question_budget=0)
    )
    message = regeneration_instruction(failed)
    assert "Do not promise absolute privacy" in message
    assert "allowed number of questions" in message
    assert message.count("Do not exceed") == 1


def test_auditor_from_metadata_never_assumes_stance_truth():
    metadata = {
        "factored_action": {
            "task_move": "clarify_goal",
            "repair_move": "clarify_one",
            "realization": {"question_budget": 1, "verbosity": "low"},
        },
        "stance": "unknown",
        "repair": {"active": True},
    }
    previous = [
        {"role": "assistant", "content": "Something else"},
        {"role": "user", "content": "The new question"},
    ]
    outcome = audit_from_generation_metadata("Statement", previous, metadata)
    assert "missing_required_clarification" in codes(outcome)


def test_audit_issues_serialization_keeps_evidence_and_counts():
    outcome = ResponseAuditor().audit("Why? Why?", context(question_budget=0))
    payload = outcome.to_dict()
    assert payload["hard_failures"] > 0
    assert isinstance(payload["issues"], list)
    assert payload["question_count"] == 2
