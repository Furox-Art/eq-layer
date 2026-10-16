"""CLI transcript parsing must not silently rewrite message roles or data."""

from __future__ import annotations

import json

import pytest

from eq_layer.cli import _load_messages, main


@pytest.mark.parametrize(
    "raw",
    [
        '[{"role":"user","role":"assistant","content":"hello"},{"role":"user","content":"next"}]',
        '[{"role":"user","content":"hello","content":"overridden"}]',
        '{"messages":[{"role":"user","content":"hello"}],"messages":[]}',
        '[{"role":"user","content":"valid","metadata":{"a":1,"a":2}}]',
        '[{"role":"user","content":"hello","content":"overwritten"}]',
    ],
)
def test_reject_duplicate_keys_at_every_depth(raw: str) -> None:
    with pytest.raises(ValueError, match="duplicate JSON"):
        _load_messages(raw)


@pytest.mark.parametrize("bad", [True, False, 3, 0.5, {}, ["hello"], None])
def test_non_string_content_is_never_coerced(bad: object) -> None:
    raw = json.dumps([{"role": "user", "content": bad}])
    with pytest.raises(ValueError, match="content must be a string"):
        _load_messages(raw)


@pytest.mark.parametrize("bad", [True, 0, {}, ["user"], None])
def test_invalid_roles_fail_as_validation_errors(bad: object) -> None:
    raw = json.dumps([{"role": bad, "content": "hello"}])
    with pytest.raises(ValueError, match="unsupported role"):
        _load_messages(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '[{"role":"user","content":"hello","metadata":NaN}]',
        '[{"role":"user","content":"hello","metadata":Infinity}]',
        '[{"role":"user","content":"hello","metadata":1e9999}]',
    ],
)
def test_reject_nonfinite_json_numbers(raw: str) -> None:
    with pytest.raises(ValueError, match="JSON"):
        _load_messages(raw)


def test_surrogate_is_not_an_accepted_transcript_character() -> None:
    with pytest.raises(ValueError, match="valid UTF-8"):
        _load_messages('[{"role":"user","content":"\\ud800"}]')


def test_multilingual_text_is_preserved_exactly() -> None:
    text = "Türkçe, 日本語 ve emoji 😀"
    payload = json.dumps([{"role": "user", "content": text}], ensure_ascii=False)
    assert _load_messages(payload) == [{"role": "user", "content": text}]


def test_bad_transcript_exits_with_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["route", "--messages-json", '[{"role":"user","content":true}]'])
    assert raised.value.code == 2
    assert "content must be a string" in capsys.readouterr().err
