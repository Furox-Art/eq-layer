"""Case files and CLI transcripts must share strict JSON interpretation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eq_layer.affect import load_cases
from eq_layer.cli import _load_messages


@pytest.mark.parametrize(
    "bad",
    [
        '{"id":"one","id":"two"}',
        '{"meta":{"label":"true","label":"false"}}',
        '{"metrics":{"score":NaN}}',
        '{"metrics":{"score":Infinity}}',
        '{"metrics":{"score":1e9999}}',
        '{"text":"\\ud800"}',
        '[1, 2, 3]',
    ],
)
def test_invalid_or_ambiguous_jsonl_case_cannot_be_loaded(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"id": "good"}) + "\n" + bad + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"cases\.jsonl:2:"):
        load_cases(str(path))


def test_valid_multilingual_cases_preserve_structure(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    cases = [{"id": "一", "label": "Türkçe"}, {"id": "β", "weights": [1.5]}]
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in cases), encoding="utf-8")
    assert load_cases(str(path)) == cases


def test_cli_transcript_remains_strict_after_shared_parser_extraction() -> None:
    with pytest.raises(ValueError, match="duplicate JSON"):
        _load_messages('[{"role":"user","role":"assistant","content":"test"}]')
    assert _load_messages('[{"role":"user","content":"Türkçe"}]') == [
        {"role": "user", "content": "Türkçe"}
    ]
