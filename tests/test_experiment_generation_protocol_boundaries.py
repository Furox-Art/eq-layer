"""Reproducible paired-arm generation with explicit no-leakage boundaries."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from eq_layer import response_experiment as exp


class FakeModel:
    model_id = "fixture"
    temperature = 0.0
    command = ("demo",)

    def __init__(self):
        self.calls = []
        self.closed = False

    def generate(self, messages, *, seed, condition):
        self.calls.append((condition, seed, messages))
        return f"answer {condition}."

    def close(self):
        self.closed = True


class FakePipeline:
    def steer_messages(self, messages, *, annotated=None):
        return (
            [{"role": "system", "content": "Preserve the user's request."}, *messages],
            {"factored_action": {"realization": {"question_budget": 1}}},
        )


def _case(identifier="c1"):
    return {
        "id": identifier,
        "source": {"stratum": "science"},
        "transcript": [{"role": "user", "content": "What is the answer?"}],
    }


def test_no_external_or_wall_clock_values_are_injected(tmp_path: Path) -> None:
    path = tmp_path / "model.bin"
    path.write_bytes(b"reproducible")
    assert exp.file_sha256(path) == hashlib.sha256(b"reproducible").hexdigest()
    assert exp._fixed_experiment_clock() == 0.0


@pytest.mark.parametrize("bad", [-1, 2, 100])
def test_regeneration_budget_must_be_predeclared(bad):
    with pytest.raises(ValueError, match="auditor_regenerations"):
        exp.generate_pairs(
            [_case()], model=FakeModel(), pipeline=FakePipeline(), auditor_regenerations=bad
        )


@pytest.mark.parametrize("bad", ["unsupported", "eq-matched"])
def test_length_profile_is_rejected_if_unknown(bad):
    with pytest.raises(ValueError, match="Unknown length profile"):
        exp.build_length_matched_messages([], {}, bad)
    with pytest.raises(ValueError, match="Unknown length profile"):
        exp.generate_pairs(
            [_case()], model=FakeModel(), pipeline=FakePipeline(), length_profiles=(bad,)
        )


def test_generation_rejects_duplicate_case_ids_and_empty_transcripts():
    model = FakeModel()
    with pytest.raises(ValueError, match="Duplicate experiment"):
        exp.generate_pairs([_case(), _case()], model=model, pipeline=FakePipeline())
    assert model.closed
    model = FakeModel()
    with pytest.raises(ValueError, match="no transcript"):
        exp.generate_pairs([{"id": "empty"}], model=model, pipeline=FakePipeline())
    assert model.closed


def test_length_control_profiles_and_eq_matched_sentence_budget():
    messages = _case()["transcript"]
    control = exp.build_length_matched_messages(messages, {"question_budget": 1}, "one_sentence")
    assert control[0]["role"] == "system"
    assert "one sentence" in control[0]["content"]
    assert control[1:] == messages

    assert exp.count_sentences("One? Two! Three.") == 3
    assert exp.count_sentences("") == 0
    matched = exp.build_eq_matched_messages(messages, {"question_budget": 1}, "One. Two.")
    assert "2 sentences" in matched[0]["content"]
    empty = exp.build_eq_matched_messages(messages, {}, "")
    assert "1 sentence" in empty[0]["content"]
    long = exp.build_eq_matched_messages(messages, {}, "A. B. C. D. E.")
    assert "4 sentences" in long[0]["content"]


def test_generation_realizability_records_all_requested_control_arms():
    model = FakeModel()
    output = exp.generate_pairs(
        [_case()],
        model=model,
        pipeline=FakePipeline(),
        seed=7,
        length_profiles=("one_sentence", exp.EQ_MATCHED_PROFILE),
    )
    assert len(output) == 1
    row = output[0]
    assert model.closed
    assert row["generation"]["same_model_both_arms"] is True
    assert row["generation"]["generated_at_epoch"] == 0.0
    assert row["generation"]["pair_seed"] == exp.stable_case_seed(7, "c1")
    assert row["generation"]["audit_regeneration_count"] == 0
    assert "length_matched_one_sentence" in row
    assert "length_matched_eq_matched" in row
    assert all(call[1] == row["generation"]["pair_seed"] for call in model.calls)


def test_legacy_length_control_arm_preserved_without_profile_override():
    model = FakeModel()
    row = exp.generate_pairs(
        [_case()], model=model, pipeline=FakePipeline(), length_matched_arm=True
    )[0]
    assert "length_matched" in row
    assert row["generation"]["order"][-1] == "length_matched"


def test_regeneration_only_when_auditor_finds_a_real_error(
    monkeypatch: pytest.MonkeyPatch,
):
    audits = [
        SimpleNamespace(should_regenerate=True, to_dict=lambda: {"status": "needs_repair"}),
        SimpleNamespace(should_regenerate=False, to_dict=lambda: {"status": "clear"}),
    ]

    def inspect(response, messages, metadata):
        return audits.pop(0)

    monkeypatch.setattr(exp, "audit_from_generation_metadata", inspect)
    monkeypatch.setattr(exp, "regeneration_instruction", lambda audit: "Correct the mistake.")
    model = FakeModel()
    row = exp.generate_pairs(
        [_case()], model=model, pipeline=FakePipeline(), auditor_regenerations=1
    )[0]
    assert row["generation"]["audit_regeneration_count"] == 1
    assert row["eq_initial"] == "answer eq."
    assert row["generation"]["response_audit"]["status"] == "clear"
    assert any(name == "eq_audit_retry" for name, _, _ in model.calls)


def test_regeneration_is_not_claimed_without_instruction(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        exp,
        "audit_from_generation_metadata",
        lambda *args: SimpleNamespace(
            should_regenerate=True, to_dict=lambda: {"status": "blocked"}
        ),
    )
    monkeypatch.setattr(exp, "regeneration_instruction", lambda audit: "")
    row = exp.generate_pairs(
        [_case()], model=FakeModel(), pipeline=FakePipeline(), auditor_regenerations=1
    )[0]
    assert row["generation"]["audit_regeneration_count"] == 0
    assert "eq_initial" not in row


def test_manifest_describes_real_sha_and_disallows_unknown_profile(tmp_path: Path):
    model = exp.CommandModel(command=("demo",), model_id="fixture")
    for name in ("affect", "subtext", "cases"):
        (tmp_path / name).write_text(name)
    args = {
        "model": model,
        "affect_model": tmp_path / "affect",
        "subtext_model": tmp_path / "subtext",
        "cases_path": tmp_path / "cases",
        "global_seed": 12,
        "n_cases": 1,
        "allow_annotations": False,
    }
    with pytest.raises(ValueError, match="Unknown length profile"):
        exp.experiment_manifest(**args, length_profiles=("invalid",))
    report = exp.experiment_manifest(**args, length_profiles=("one_sentence",))
    assert report["cases_sha256"] == exp.file_sha256(tmp_path / "cases")
    assert report["n_cases"] == 1
    assert "length_matched:one_sentence" in report["arms"]
