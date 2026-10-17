"""Reference contracts for preference inference and blinded experiment intake."""

from __future__ import annotations

import pytest

from eq_layer.preferences import SessionPreferences, infer_session_preferences
from eq_layer.response_experiment import (
    select_stratified_cases,
    stable_case_seed,
    stratum_counts,
    validate_experiment_cases,
)


def _message(role: str, text: object) -> dict:
    return {"role": role, "content": text}


def test_session_preference_inference_tracks_only_explicit_user_scope() -> None:
    state = infer_session_preferences(
        [
            _message("assistant", "From now on respond brief and direct."),
            _message("user", "Brief answer, please."),  # one-off, not a session request
            _message("user", "From now on, keep responses concise and direct. Don't guess."),
            _message("user", "Throughout this conversation, respond in detail."),
            _message("user", "Only answer what I ask throughout this conversation."),
            _message("assistant", "Stay within scope."),
        ]
    )
    assert isinstance(state, SessionPreferences)
    assert state.verbosity == "high"
    assert state.directness == "high"
    assert state.no_guess
    assert state.scope_limited
    assert state.source_turns == (2, 3, 4)
    assert len(state.evidence) >= 4
    assert state.to_dict()["source_turns"] == (2, 3, 4)


@pytest.mark.parametrize(
    ("statement", "field", "expected"),
    [
        ("Benimle kısa ve net konuş", "verbosity", "low"),
        ("Her zaman detaylı cevap ver", "verbosity", "high"),
        ("Bu sohbette direkt yanıt ver", "directness", "high"),
        ("For this conversation, don't guess", "no_guess", True),
        ("Always stay within scope", "scope_limited", True),
        ("Make your responses concise", "verbosity", "low"),
        ("Respond briefly from now on", "verbosity", "low"),
    ],
)
def test_preferences_recognize_explicit_scoped_statements(
    statement: str, field: str, expected: object
) -> None:
    state = infer_session_preferences([_message("user", statement)])
    assert getattr(state, field) == expected


def test_preferences_do_not_persist_generic_or_empty_turns() -> None:
    result = infer_session_preferences(
        [
            _message("user", ""),
            _message("user", "What's the weather?"),
            _message("system", "Always guess"),
            _message("user", "Every day I read detailed material"),
            _message("user", None),
        ]
    )
    assert result == SessionPreferences()


def _case(identifier: str, stratum: str = "science") -> dict:
    return {
        "id": identifier,
        "source": {"stratum": stratum},
        "transcript": [{"role": "user", "content": "Question?"}],
    }


def test_experiment_intake_validates_roles_ids_and_gold_label_leakage() -> None:
    cases = [
        _case("one"),
        {**_case("one"), "expected_policy": "gold"},
        {"id": "", "transcript": [_message("user", "hey")]},
        {"id": "no-context"},
        {"id": "no-user", "context": [_message("assistant", "response")]},
        {"id": "broken", "transcript": [None, {"role": "tool", "content": ""}]},
    ]
    report = validate_experiment_cases(cases, final=True)
    assert not report["ok"]
    joined = "\n".join(report["errors"])
    for part in (
        "duplicate",
        "missing id",
        "transcript/context",
        "no user",
        "non-object",
        "empty message",
        "gold",
        "at least 120",
    ):
        assert part in joined
    assert report["n_cases"] == 6
    assert report["role_counts"]["other"] >= 1


def test_experiment_final_120_cases_satisfy_protocol() -> None:
    cases = [_case(f"c-{i}", "math" if i % 2 else "science") for i in range(120)]
    report = validate_experiment_cases(cases, final=True)
    assert report["ok"]
    assert not report["errors"]
    assert report["unique_ids"] == 120


@pytest.mark.parametrize("limit", [0, -1])
def test_stratified_selector_rejects_empty_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        select_stratified_cases([_case("first")], limit)


def test_stratified_selector_rejects_missing_strata_and_empty_corpus() -> None:
    with pytest.raises(ValueError, match="source.stratum"):
        select_stratified_cases([{"id": "case", "source": {}}], 1)
    with pytest.raises(ValueError, match="No strata"):
        select_stratified_cases([], 1)


def test_stratified_selector_balances_and_redistributes_small_strata() -> None:
    cases = [_case("A0", "A")] + [_case(f"B{i}", "B") for i in range(6)]
    picked = select_stratified_cases(cases, 6)
    assert len(picked) == 6
    assert picked[0]["id"] == "A0"
    assert [row["id"] for row in picked[1:]] == [f"B{i}" for i in range(5)]
    assert stratum_counts(picked) == {"A": 1, "B": 5}
    assert select_stratified_cases(cases, 100) == cases


def test_stratified_selector_handles_multiple_equal_strata_round_robin() -> None:
    cases = [
        _case("B1", "B"),
        _case("A1", "A"),
        _case("B2", "B"),
        _case("A2", "A"),
    ]
    assert [r["id"] for r in select_stratified_cases(cases, 3)] == ["A1", "B1", "A2"]


def test_stratum_inventory_marks_missing_source_as_unstratified() -> None:
    assert stratum_counts([_case("x"), {"id": "y"}, {"id": "z", "source": "invalid"}]) == {
        "science": 1,
        "unstratified": 2,
    }


def test_case_seed_is_stable_and_separates_case_identifiers() -> None:
    assert stable_case_seed(42, "a") == stable_case_seed(42, "a")
    assert stable_case_seed(42, "a") != stable_case_seed(42, "b")
    assert 0 <= stable_case_seed(42, "a") < 2**32
