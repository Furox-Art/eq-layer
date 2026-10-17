"""Small real model training plus controlled signal-to-subtext decision gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eq_layer import trained_intent, trained_subtext
from eq_layer.policies import AffectState


def _affect(valence=0.0, arousal=0.0, *, history=()):
    return AffectState(
        valence=valence,
        arousal=arousal,
        escalation_delta=0.0,
        stance="unknown",
        subtext="statement",
        turn_index=0,
        history=history,
    )


def _signals(act="inform", emotion="neutral", act_confidence=0.95, emotion_confidence=0.95):
    return trained_subtext.DialogueSignals(act, emotion, act_confidence, emotion_confidence)


@pytest.mark.parametrize(
    ("act", "emotion", "arousal", "valence", "wanted"),
    [
        ("directive", "anger", 0.1, 0, "demand"),
        ("directive", "neutral", 0.6, 0, "demand"),
        ("question", "anger", 0.6, 0, "challenge"),
        ("question", "neutral", 0.1, 0, "question"),
        ("inform", "sadness", 0.1, -0.6, "exhaustion"),
        ("inform", "neutral", 0.1, 0, "statement"),
        ("commissive", "neutral", 0.1, 0, "statement"),
    ],
)
def test_learned_subtext_routes_direct_and_derived_signals(
    monkeypatch: pytest.MonkeyPatch,
    act,
    emotion,
    arousal,
    valence,
    wanted,
) -> None:
    model = trained_subtext.TrainedSubtext()
    monkeypatch.setattr(model, "predict_signals", lambda text: _signals(act, emotion))
    state = model.infer(
        [{"role": "user", "content": "Neutral content with no marker"}],
        _affect(valence, arousal),
    )
    assert state.label == wanted
    assert state.signals is not None
    assert state.evidence_level in {"derived", "direct_learned"}
    assert state.confidence == pytest.approx(0.95)


def test_drained_two_step_history_changes_exhaustion_to_resignation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = trained_subtext.TrainedSubtext()
    monkeypatch.setattr(model, "predict_signals", lambda _: _signals("inform", "sadness"))
    calm_negative = _affect(-0.7, 0.2)
    history = (calm_negative, calm_negative)
    state = model.infer(
        [{"role": "user", "content": "Neutral words"}],
        _affect(-0.7, 0.2, history=history),
    )
    assert state.label == "resignation"
    assert model._drained_run(_affect(-0.7, 0.2, history=history))
    assert not model._drained_run(_affect(-0.7, 0.2, history=(calm_negative,)))
    assert not model._drained_run(_affect(-0.7, 0.2, history=(_affect(+0.2, 0.8), calm_negative)))


@pytest.mark.parametrize(
    ("messages", "annotation", "wanted"),
    [
        ([{"role": "user", "content": "Neutral"}], {"subtext": "manual"}, "manual"),
        ([], {}, "statement"),
        ([{"role": "assistant", "content": "Only assistant"}], {}, "statement"),
        (
            [
                {"role": "assistant", "content": "Wrong value"},
                {"role": "user", "content": "No, that is wrong"},
            ],
            {},
            "correction",
        ),
        ([{"role": "user", "content": "Gizli tut"}], {}, "disclosure_request"),
        (
            [
                {"role": "user", "content": "Neden bu?"},
                {"role": "user", "content": "Neden yine?"},
                {"role": "user", "content": "Neden şimdi?"},
            ],
            {},
            "escalating",
        ),
        (
            [
                {"role": "user", "content": "Yap!"},
                {"role": "user", "content": "Yap!"},
                {"role": "user", "content": "Yap!"},
            ],
            {},
            "demand",
        ),
    ],
)
def test_structural_subtext_priority_bypasses_inappropriate_ml_predictions(
    messages,
    annotation,
    wanted,
) -> None:
    model = trained_subtext.TrainedSubtext()
    result = model.infer(messages, _affect(), annotated=annotation)
    assert result.label == wanted
    assert result.signals is None


def test_subtext_rejects_weak_confidence_instead_of_inventing_a_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = trained_subtext.TrainedSubtext()
    monkeypatch.setattr(
        model,
        "predict_signals",
        lambda _: _signals("question", "anger", 0.1, 0.1),
    )
    response = model.infer([{"role": "user", "content": "Plain words"}], _affect())
    assert response.evidence_level == "fallback"
    assert response.confidence == 0.1
    assert response.label == "statement"


def test_inference_raises_when_subtext_pipeline_is_not_trained() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        trained_subtext.TrainedSubtext().predict_signals("text")
    with pytest.raises(RuntimeError, match="not fitted"):
        trained_subtext.TrainedSubtext().save("missing.joblib")
    with pytest.raises(ValueError, match="at least one record"):
        trained_subtext.TrainedSubtext().fit([])


def _subtext_rows():
    return [
        {
            "text": f"This is example message {i} "
            + ("Please answer my question" if i % 2 else "I have information"),
            "act": "question" if i % 2 else "inform",
            "emotion": "anger" if i % 2 else "neutral",
        }
        for i in range(18)
    ]


def test_real_subtext_two_class_fit_predict_and_joblib_roundtrip(tmp_path: Path) -> None:
    model = trained_subtext.TrainedSubtext()
    assert model.fit(_subtext_rows()) is model
    signal = model.predict_signals("Please answer my question")
    assert signal.act in {"inform", "question"}
    assert signal.emotion in {"anger", "neutral"}
    assert 0 <= signal.act_confidence <= 1
    artifact = tmp_path / "subtext.joblib"
    model.save(artifact)
    restored = trained_subtext.TrainedSubtext.load(artifact)
    assert restored.training_examples == len(_subtext_rows())
    assert restored.predict_signals("Please answer my question") == signal


def test_intent_rejects_one_class_and_unfitted_model() -> None:
    with pytest.raises(ValueError, match="at least two labels"):
        trained_intent.TrainedIntent().fit([{"text": "x", "label": "statement"}])
    with pytest.raises(RuntimeError, match="not fitted"):
        trained_intent.TrainedIntent().infer([{"role": "user", "content": "Hi"}])
    with pytest.raises(RuntimeError, match="not fitted"):
        trained_intent.TrainedIntent().save("not-fitted.joblib")


def test_real_intent_training_roundtrip_and_empty_turn(tmp_path: Path) -> None:
    records = [
        {
            "text": f"example {i} "
            + ("Please explain the theorem" if i % 2 else "That is a statement"),
            "label": "explanation" if i % 2 else "statement",
        }
        for i in range(12)
    ]
    training = tmp_path / "intent.jsonl"
    training.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    assert len(trained_intent.load_jsonl_records(training)) == 12
    model = trained_intent.TrainedIntent.from_jsonl(training)
    assert model.training_examples == 12
    state = model.infer([{"role": "user", "content": "Please explain the theorem"}])
    assert 0 <= state.confidence <= 1
    assert state.belief is not None
    assert len(state.evidence) == 6
    empty = model.infer([])
    assert empty.kind == "unknown"
    assert empty.needs_clarification
    assert model.reconcile_with_dialogue_evidence(empty, [], None) == empty

    artifact = tmp_path / "intent.joblib"
    model.save(artifact)
    restored = trained_intent.TrainedIntent.load(artifact)
    assert restored.training_examples == 12
    assert restored.infer([{"role": "user", "content": "Please explain the theorem"}]) == state
