from eq_layer.policies import Register
from eq_layer.steer import (
    aggregate,
    escalation_latency,
    genericness_score,
    score_case,
    specificity_score,
)


def test_specificity_is_concept_matched_and_never_estimated():
    assert specificity_score("unrelated", {}) == 0.0
    assert specificity_score("alpha and beta", {"expected_concepts": ["alpha", "beta"]}) == 1.0
    assert specificity_score("alpha", {"expected_concepts": ["alpha", "beta"]}) == 0.5


def test_generic_phrase_penalty_is_bounded():
    assert genericness_score("I hear you") == 0.5
    assert genericness_score("I hear you. Your feelings are valid") == 1.0


def test_escalation_latency_never_invents_unobserved_transition():
    assert escalation_latency([Register.MIRROR.value] * 3) is None
    assert escalation_latency([Register.MIRROR.value, "direct"]) == 1
    assert escalation_latency([]) is None


def test_composite_scoring_and_aggregation_round_trip():
    item = score_case(
        "alpha and beta",
        {"id": "test", "expected_concepts": ["alpha", "beta"]},
        [Register.MIRROR.value, "direct"],
    )
    assert item["escalation_latency"] == 1
    assert item["specificity"] == 1.0
    assert 0 <= item["composite"] <= 1
    assert aggregate([]) == {"n": 0}
    report = aggregate([item, {**item, "escalation_latency": None}])
    assert report["n"] == 2
    assert report["unresolved_escalations"] == 1


def test_empty_evidence_is_zero_quality_not_success():
    item = score_case("", {"id": "empty"})
    assert item["specificity"] == 0.0
    assert item["escalation_latency"] is None
    report = aggregate([item])
    assert report["specificity"] == 0.0
    assert report["unresolved_escalations"] == 1
