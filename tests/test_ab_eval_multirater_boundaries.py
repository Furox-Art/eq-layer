import pytest
from eq_layer import ab_eval as ab


def row(i, vote):
    return {"id": i, "ratings": {d: vote for d in ab.DIMENSIONS}}


def keys(*ids):
    return {i: {"A": "eq", "B": "baseline"} for i in ids}


def test_kappa_invalid_and_degenerate_samples_are_unknown():
    assert ab._fleiss_kappa([]) is None
    assert ab._fleiss_kappa([["eq"]]) is None
    assert ab._fleiss_kappa([["eq", "baseline"], ["eq"]]) is None
    assert ab._fleiss_kappa([["eq", "eq"]]) is None
    assert ab._fleiss_kappa([["eq", "baseline"], ["baseline", "eq"]]) == -1.0


def test_multi_rater_rejects_single_rater_and_nonunique_cases():
    with pytest.raises(ValueError, match="at least two raters"):
        ab.score_multiple_blinded({"a": [row("case", "A")]}, keys("case"))
    with pytest.raises(ValueError, match="Duplicate case"):
        ab.score_multiple_blinded(
            {
                "a": [row("case", "A"), row("case", "B")],
                "b": [row("case", "A")],
            },
            keys("case"),
        )


def test_multi_rater_requires_identical_sample_sets():
    with pytest.raises(ValueError, match="same case ids"):
        ab.score_multiple_blinded(
            {
                "a": [row("one", "A")],
                "b": [row("two", "A")],
            },
            keys("one", "two"),
        )


def test_multi_rater_majority_and_tie_are_not_counted_as_unique_raters():
    report = ab.score_multiple_blinded(
        {
            "a": [row("one", "A"), row("two", "B")],
            "b": [row("one", "B"), row("two", "B")],
        },
        keys("one", "two"),
    )
    assert report["n_cases"] == 2 and report["n_raters"] == 2
    result = report["dimensions"]["overall"]
    assert result["case_majority_ties"] == 1
    assert result["case_majority_eq_losses"] == 1
    assert result["fleiss_kappa"] is not None
