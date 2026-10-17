import json
from pathlib import Path

import pytest

from eq_layer import ab_eval as ab


def pair(identifier="c1"):
    return {"id": identifier, "baseline": "control reply", "eq": "eq response"}


def test_reproducible_blinding_and_duplicate_guard():
    a, key = ab.prepare_blinded([pair("a"), pair("b")], seed=101)
    b, key_b = ab.prepare_blinded([pair("a"), pair("b")], seed=101)
    assert a == b and key == key_b
    assert all(set(item.values()) == {"eq", "baseline"} for item in key.values())
    with pytest.raises(ValueError, match="Duplicate A/B case"):
        ab.prepare_blinded([pair(), pair()])


def test_jsonl_writer_reader_accepts_empty_lines_not_ambiguous_json(tmp_path: Path):
    path = tmp_path / "ballots.jsonl"
    ab.write_jsonl(path, [pair()])
    assert ab.load_jsonl(path) == [pair()]
    path.write_text("\n \n" + json.dumps(pair()) + "\n")
    assert ab.load_jsonl(path) == [pair()]
    path.write_text('{"id":"1","id":"2"}\n')
    with pytest.raises(ValueError, match="invalid evaluation JSON"):
        ab.load_jsonl(path)
    path.write_text("[]\n")
    with pytest.raises(ValueError, match="evaluation row must be an object"):
        ab.load_jsonl(path)


def test_wilson_and_binomial_reject_zero_total():
    with pytest.raises(ValueError, match="total > 0"):
        ab._wilson_interval(0, 0)
    with pytest.raises(ValueError, match="total > 0"):
        ab._binom_two_sided_half(0, 0)
    assert ab._binom_two_sided_half(0, 2) == 0.5
    assert ab.summarize(0, 0, 2).exact_p_two_sided is None
    assert ab.summarize(2, 1, 0).exact_p_two_sided is not None
