"""Prevent fabricated A/B responses and inference over an empty sample."""

from __future__ import annotations

import pytest

from eq_layer.ab_eval import DIMENSIONS, prepare_blinded, score_multiple_blinded


@pytest.mark.parametrize("bad_id", [None, 0, 1, True, "", "  ", []])
def test_blinding_rejects_non_text_or_blank_case_ids(bad_id: object) -> None:
    with pytest.raises(ValueError, match="case id must be a non-empty string"):
        prepare_blinded([{"id": bad_id, "baseline": "control", "eq": "treatment"}])


@pytest.mark.parametrize("field", ["eq", "baseline"])
@pytest.mark.parametrize("value", [None, True, False, 42, "", " ", []])
def test_blinding_refuses_fake_or_empty_answer_strings(field: str, value: object) -> None:
    pair = {"id": "case-1", "baseline": "control", "eq": "treatment"}
    pair[field] = value
    with pytest.raises(ValueError, match="responses must be non-empty strings"):
        prepare_blinded([pair])


@pytest.mark.parametrize("source", ["metadata", ["stratum"], 42])
def test_blinding_rejects_invalid_source_metadata(source: object) -> None:
    pair = {"id": "case-1", "baseline": "control", "eq": "treatment", "source": source}
    with pytest.raises(ValueError, match="source must be an object"):
        prepare_blinded([pair])


def test_blinding_preserves_multilingual_responses_and_metadata() -> None:
    cases = [{
        "id": "Türkçe", "baseline": "kontrol", "eq": "日本語 😀",
        "source": {"stratum": "science"}, "context": [],
    }]
    ballot, key = prepare_blinded(cases, seed=1)
    assert ballot[0]["id"] == "Türkçe"
    assert {ballot[0]["response_A"], ballot[0]["response_B"]} == {"kontrol", "日本語 😀"}
    assert ballot[0]["stratum"] == "science"
    assert set(key["Türkçe"].values()) == {"eq", "baseline"}


def test_multi_rater_inference_refuses_zero_case_experiment() -> None:
    with pytest.raises(ValueError, match="at least one rated case"):
        score_multiple_blinded({"rater-1": [], "rater-2": []}, {})


def test_multi_rater_nonempty_experiment_remains_compatible() -> None:
    case = {"id": "case-1", "ratings": {dim: "A" for dim in DIMENSIONS}}
    result = score_multiple_blinded(
        {"rater-1": [case], "rater-2": [case]},
        {"case-1": {"A": "eq", "B": "baseline"}},
    )
    assert result["n_cases"] == 1
    assert result["dimensions"]["overall"]["case_majority_eq_wins"] == 1
