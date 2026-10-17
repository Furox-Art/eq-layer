"""Small reproducible scikit-learn training runs and artifact round trips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from eq_layer import trained_affect as affect
from eq_layer import trained_breakdown as breakdown
from eq_layer import trained_disagreement as disagreement
from eq_layer import trained_subtext as subtext


@dataclass
class BreakdownSample:
    model_text: str
    hard_breakdown: bool


@dataclass
class DisagreementSample:
    model_text: str
    label: str


def _breakdown_rows() -> list[BreakdownSample]:
    return [
        BreakdownSample(
            f"Repeated conversation {i} "
            + (
                "The system misunderstood me and stopped answering"
                if i % 2
                else "Thank you for that useful answer, I understand"
            ),
            bool(i % 2),
        )
        for i in range(220)
    ]


def _disagreement_rows() -> list[DisagreementSample]:
    return [
        DisagreementSample(
            f"Reply number {i}: "
            + ("I disagree with your assertion" if i % 2 else "I agree with your proposal"),
            "disagreement" if i % 2 else "agreement",
        )
        for i in range(220)
    ]


def test_real_breakdown_pipeline_trains_predicts_and_roundtrips(tmp_path: Path) -> None:
    model = breakdown.TrainedBreakdown()
    assert model.fit(iter(_breakdown_rows())) is model
    assert model.training_examples == 220
    assert model.positive_examples == 110
    with_context = model.predict("I need help", ["Earlier question", "", "    "])
    without_context = model.predict("I need help")
    for value in (with_context, without_context):
        assert isinstance(value, breakdown.BreakdownDecision)
        assert 0 <= value.probability <= 1
        assert len(value.evidence) == 2

    artifact = tmp_path / "breakdown.joblib"
    model.save(artifact)
    restored = breakdown.TrainedBreakdown.load(artifact)
    assert restored.positive_examples == model.positive_examples
    assert restored.training_examples == model.training_examples
    assert restored.predict("I need help", ["Earlier question"]) == with_context


def test_real_disagreement_pipeline_trains_predicts_and_roundtrips(tmp_path: Path) -> None:
    model = disagreement.TrainedDisagreement()
    assert model.fit(iter(_disagreement_rows())) is model
    assert model.training_examples == 220
    assert model.positive_examples == 110
    in_reply = model.predict("Your result is wrong", "I believe the result")
    bare_reply = model.predict("Your result is wrong")
    for value in (in_reply, bare_reply):
        assert isinstance(value, disagreement.DisagreementDecision)
        assert 0 <= value.probability <= 1
        assert len(value.evidence) == 2
    artifact = tmp_path / "disagreement.joblib"
    model.save(artifact)
    restored = disagreement.TrainedDisagreement.load(artifact)
    assert restored.positive_examples == 110
    assert restored.predict("Your result is wrong", "I believe the result") == in_reply


def test_training_rejects_unusable_or_unbalanced_labels() -> None:
    for model in (breakdown.TrainedBreakdown(), disagreement.TrainedDisagreement()):
        with pytest.raises(ValueError, match="at least one record"):
            model.fit([])
    with pytest.raises(ValueError, match="Too few hard-breakdown"):
        breakdown.TrainedBreakdown().fit(_breakdown_rows()[:10])
    with pytest.raises(ValueError, match="Too few disagreement"):
        disagreement.TrainedDisagreement().fit(_disagreement_rows()[:10])


def test_unfitted_model_calls_fail_explicitly(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        breakdown.TrainedBreakdown().predict("text")
    with pytest.raises(RuntimeError, match="not fitted"):
        disagreement.TrainedDisagreement().predict("text")
    with pytest.raises(RuntimeError, match="not fitted"):
        breakdown.TrainedBreakdown().save(tmp_path / "unused")
    with pytest.raises(RuntimeError, match="not fitted"):
        disagreement.TrainedDisagreement().save(tmp_path / "unused")
    with pytest.raises(RuntimeError, match="not fitted"):
        affect.TrainedAffect().save(tmp_path / "unused")


def test_emobank_csv_parser_validates_required_columns_and_rows(tmp_path: Path) -> None:
    malformed = tmp_path / "no-scores.csv"
    malformed.write_text("split,text\ntrain,hello\n")
    with pytest.raises(ValueError, match="missing required columns"):
        affect.load_emobank(malformed)

    csv_path = tmp_path / "emobank.csv"
    csv_path.write_text(
        "split,V,A,D,text\n"
        "train,4.0,2.0,3.5,That was helpful\n"
        "test,2.0,4.0,2.5,I feel concerned\n"
        "train,3.0,3.0,3.0, \n"
        "train,5.0,5.0,5.0,Thank you, very helpful\n"
    )
    rows = affect.load_emobank(csv_path)
    assert len(rows) == 3
    assert len(affect.load_emobank(csv_path, split="train")) == 2
    assert affect.load_emobank(csv_path, split="dev") == []


def test_real_affect_training_inference_and_artifact_provenance(tmp_path: Path) -> None:
    records = [
        {"text": "I am very glad", "V": 5.0, "A": 4.0, "D": 4.0},
        {"text": "I am truly sad", "V": 1.0, "A": 2.0, "D": 2.0},
        {"text": "I am quite glad", "V": 4.0, "A": 3.0, "D": 3.0},
        {"text": "I feel sad now", "V": 2.0, "A": 2.0, "D": 2.0},
    ]
    model = affect.TrainedAffect()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_dimensional("test")
    with pytest.raises(RuntimeError, match="not fitted"):
        model.infer([])
    with pytest.raises(ValueError, match="at least one"):
        model.fit([])
    assert model.fit(records) is model
    dims = model.predict_dimensional("I feel glad")
    assert all(1 <= x <= 5 for x in (dims.valence, dims.arousal, dims.dominance))
    empty = model.infer([], turn_index=7)
    assert empty.turn_index == 7
    assert empty.history == ()
    inferred = model.infer(
        [
            {"role": "user", "content": "I am glad"},
            {"role": "assistant", "content": "Fine."},
            {"role": "user", "content": "I feel sad"},
        ],
        turn_index=6,
    )
    assert len(inferred.history) == 2
    assert inferred.turn_index == 6
    assert -1 <= inferred.valence <= 1
    assert 0 <= inferred.arousal <= 1

    artifact = tmp_path / "affect.joblib"
    model.save(artifact)
    restored = affect.TrainedAffect.load(artifact)
    assert restored.training_examples == 4
    assert restored.training_source == model.training_source
    assert restored.training_license == model.training_license
    assert restored.predict_dimensional("I feel glad") == dims


def test_affect_normalization_and_clamping_edge_cases() -> None:
    assert affect.TrainedAffect._normalise_valence(1) == -1
    assert affect.TrainedAffect._normalise_valence(5) == 1
    assert affect.TrainedAffect._normalise_valence(-100) == -1
    assert affect.TrainedAffect._normalise_arousal(-100) == 0
    assert affect.TrainedAffect._normalise_arousal(100) == 1
    assert affect.TrainedAffect._clamp(12.0, 1, 5) == 5


def test_subtext_parser_rejects_misaligned_utterances(tmp_path: Path) -> None:
    data = tmp_path / "invalid.txt"
    data.write_text("Hello __eou__ Bye __eou__\tDaily\t1\t0 1\n")
    with pytest.raises(ValueError, match="alignment"):
        subtext.load_xdailydialog(data)
    data.write_text("Hello __eou__\tDaily\t99\t0\n")
    with pytest.raises(ValueError, match="act code"):
        subtext.load_xdailydialog(data)
    data.write_text("Hello __eou__\tDaily\t1\t99\n")
    with pytest.raises(ValueError, match="emotion code"):
        subtext.load_xdailydialog(data)
    data.write_text("Hello __eou__\tDaily\t1\n")
    with pytest.raises(ValueError, match="expected 4"):
        subtext.load_xdailydialog(data)


def test_subtext_parses_official_four_field_annotation(tmp_path: Path) -> None:
    data = tmp_path / "sample.txt"
    data.write_text("\nHello __eou__ Thank you __eou__\tDaily\t1 2\t0 4\n")
    rows = subtext.load_xdailydialog(data)
    assert [row["act"] for row in rows] == ["inform", "question"]
    assert [row["emotion"] for row in rows] == ["neutral", "happiness"]
    assert [row["turn_index"] for row in rows] == [0, 1]
