"""External A/B ballots must be uniquely decoded, not silently counted twice."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eq_layer.ab_eval import (
    DIMENSIONS,
    load_jsonl,
    prepare_blinded,
    score_blinded,
    score_multiple_blinded,
)


def _row(case_id: str, vote: str = "A") -> dict:
    return {
        "id": case_id,
        "ratings": {dimension: vote for dimension in DIMENSIONS},
    }


@pytest.mark.parametrize(
    "bad",
    [
        '{"id":"one","id":"two"}',
        '{"nested":{"id":"one","id":"two"}}',
        '{"score":NaN}',
        '{"score":Infinity}',
        '{"score":1e9999}',
        r'{"text":"\ud800"}',
        '["not", "an", "object"]',
    ],
)
def test_ab_jsonl_rejects_ambiguous_rows_with_source_line(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "votes.jsonl"
    path.write_text('{"id":"valid"}\n' + bad + '\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"votes\.jsonl:2:"):
        load_jsonl(path)


def test_valid_multilingual_ab_jsonl_keeps_rows(tmp_path: Path) -> None:
    path = tmp_path / "votes.jsonl"
    rows = [{"id": "Türkçe", "answer": "日本語 😀"}, {"id": "β", "votes": 2}]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    assert load_jsonl(path) == rows


def test_repeated_ballot_id_does_not_duplicate_significance_trials() -> None:
    key = {"case-1": {"A": "eq", "B": "baseline"}}
    with pytest.raises(ValueError, match="Duplicate case id"):
        score_blinded([_row("case-1"), _row("case-1")], key)


def test_repeated_excluded_id_is_still_invalid() -> None:
    with pytest.raises(ValueError, match="Duplicate case id"):
        score_blinded([_row("case-1"), _row("case-1")], {}, exclude_ids={"case-1"})


@pytest.mark.parametrize(
    "mapping",
    [
        {"A": "eq", "B": "eq"},
        {"A": "baseline", "B": "baseline"},
        {"A": "eq"},
        {"A": "eq", "B": "baseline", "C": "eq"},
    ],
)
def test_single_rater_cannot_score_corrupted_decode_key(mapping: dict) -> None:
    with pytest.raises(ValueError, match="non-blinded decode key"):
        score_blinded([_row("case-1")], {"case-1": mapping})


def test_multi_rater_also_rejects_corrupted_decode_key() -> None:
    ballot = [_row("case-1")]
    with pytest.raises(ValueError, match="non-blinded decode key"):
        score_multiple_blinded(
            {"a": ballot, "b": ballot}, {"case-1": {"A": "eq", "B": "eq"}}
        )


def test_legitimate_blinding_and_votes_keep_expected_count() -> None:
    original = [{"id": "case-1", "baseline": "control", "eq": "treatment"}]
    ballot, key = prepare_blinded(original, seed=9)
    for row in ballot:
        eq_side = "A" if key[row["id"]]["A"] == "eq" else "B"
        row["ratings"] = {dimension: eq_side for dimension in DIMENSIONS}
    report = score_blinded(ballot, key)
    assert report["dimensions"]["overall"]["eq_wins"] == 1
    assert report["dimensions"]["overall"]["eq_losses"] == 0
