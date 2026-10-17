import pytest
from eq_layer import ab_eval as ab


def vote(identifier, response):
    return {"id": identifier, "ratings": {d: response for d in ab.DIMENSIONS}}


def key(*ids):
    return {i: {"A": "eq", "B": "baseline"} for i in ids}


@pytest.mark.parametrize(
    "invalid", [{}, {"a": {"A": "eq"}}, {"a": {"A": "eq", "B": "eq"}}, {"a": ["eq", "baseline"]}]
)
def test_missing_or_nonblind_decoding_is_rejected(invalid):
    with pytest.raises(ValueError):
        ab._validate_decode_arms("a", invalid)


@pytest.mark.parametrize(
    "raw,expected", [("A", "eq_wins"), ("B", "eq_losses"), ("TIE", "ties"), ("tie", "ties")]
)
def test_blinded_tally_counts_case_level_votes(raw, expected):
    result = ab.score_blinded([vote("a", raw)], key("a"))
    stats = result["dimensions"]["overall"]
    assert stats[expected] == 1
    assert sum(stats[field] for field in ("eq_wins", "eq_losses", "ties")) == 1


def test_bad_vote_and_duplicate_case_are_errors():
    with pytest.raises(ValueError, match="expected A, B, or tie"):
        ab.score_blinded([vote("a", "?")], key("a"))
    with pytest.raises(ValueError, match="Duplicate case"):
        ab.score_blinded([vote("a", "A"), vote("a", "B")], key("a"))


def test_exclusions_are_stated_not_silently_scored():
    with pytest.raises(ValueError, match="Missing decode key"):
        ab.score_blinded([vote("a", "A")], {})
    report = ab.score_blinded([vote("a", "A")], {}, exclude_ids={"a"})
    assert report["n_excluded"] == 1 and report["excluded_ids"] == ["a"]


def test_decoding_keeps_ties_and_rejects_bad_labels():
    assert ab._decoded_vote(vote("a", "A"), key("a"), "overall") == "eq"
    assert ab._decoded_vote(vote("a", "B"), key("a"), "overall") == "baseline"
    assert ab._decoded_vote(vote("a", "tie"), key("a"), "overall") == "tie"
    with pytest.raises(ValueError):
        ab._decoded_vote(vote("a", "?"), key("a"), "overall")
