"""Classifier determinism with stub artifacts (hardening round 6).

The learned path needs three fitted estimators: an intent classifier
(TF-IDF + logistic regression), an affect regressor over EmoBank V/A/D and a
subtext signal pair over XDailyDialog dialogue acts and emotions. scikit-learn
is an optional extra and is not installed here, so every one of them is
replaced by a deterministic stub fitted estimator loaded through the *real*
artifact path (a stubbed ``joblib``), which is exactly the code path that
audits asked to hold still.

Two properties are asserted:

* repeat invocation of the same rater, and of two independently loaded
  pipelines, produce byte-identical output;
* the order the cases are routed in does not change any decision.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from eq_layer.intent_belief import INTENT_LABELS
from eq_layer.response_experiment import FullEQPipeline
from eq_layer.trained_affect import TrainedAffect
from eq_layer.trained_intent import TrainedIntent
from eq_layer.trained_subtext import ACT_MAP, EMOTION_MAP

AFFECT_ARTIFACT = Path("artifacts/affect.joblib")
SUBTEXT_ARTIFACT = Path("artifacts/subtext.joblib")

ACT_LABELS = tuple(ACT_MAP.values())
EMOTION_LABELS = tuple(EMOTION_MAP.values())

TRANSCRIPTS: tuple[tuple[tuple[str, str], ...], ...] = (
    (("user", "Bu raporu yarın sabah gönder"),),
    (("user", "Neden bu kadar yavaş?"), ("assistant", "Bakıyorum"), ("user", "Neden yine?")),
    (
        ("user", "Sunumu erteledim"),
        ("assistant", "Sorun değil"),
        ("user", "Boş verdim"),
    ),
    (("assistant", "You said version 3."), ("user", "I said version 4")),
    (("user", "Randevumu üç kez değiştirdiler"),),
    (("user", "Bunu kimseye anlatma"),),
)

MESSAGES = tuple(
    [{"role": role, "content": text} for role, text in turns] for turns in TRANSCRIPTS
)


def _deterministic_weights(text: str, count: int, offset: int) -> list[float]:
    """A fixed, text-only derived weight vector. No clock, no RNG, no state."""
    base = sum(ord(char) for char in text)
    return [1.0 + (base * (index + 3) + offset) % 23 for index in range(count)]


class _StubProbabilityRow:
    """Minimal stand-in for a numpy probability row."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, index: int) -> float:
        return self._values[index]

    def argmax(self) -> int:
        return max(range(len(self._values)), key=lambda index: self._values[index])


class StubIntentClassifier:
    """Deterministic stand-in for the fitted intent classifier."""

    def __init__(self) -> None:
        self.classes_ = list(INTENT_LABELS)

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        rows = []
        for index, text in enumerate(texts):
            weights = _deterministic_weights(f"{index}:{text}", len(self.classes_), 3)
            total = sum(weights)
            rows.append([weight / total for weight in weights])
        return rows


class StubAffectRegressor:
    """Deterministic stand-in for the fitted EmoBank V/A/D regressor."""

    def predict(self, texts: list[str]) -> list[tuple[float, float, float]]:
        rows = []
        for index, text in enumerate(texts):
            weights = _deterministic_weights(f"{index}:{text}", 3, 11)
            rows.append(tuple(1.0 + (weight % 5) * 0.5 for weight in weights))  # type: ignore[return-value]
        return rows


class StubFeatureExtractor:
    """Deterministic stand-in for the TF-IDF feature union."""

    def transform(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(len(text)), float(sum(ord(char) for char in text) % 7)]
            for text in texts
        ]


class StubSignalClassifier:
    """Deterministic stand-in for a dialogue-act/emotion classifier."""

    def __init__(self, labels: tuple[str, ...], offset: int) -> None:
        self.classes_ = list(labels)
        self._offset = offset

    def predict_proba(self, matrix: list[list[float]]) -> list[_StubProbabilityRow]:
        rows = []
        for row in matrix:
            weights = [
                1.0 + (int(row[0]) * self._offset + index * (int(row[1]) + 1)) % 13
                for index in range(len(self.classes_))
            ]
            total = sum(weights)
            rows.append(_StubProbabilityRow([weight / total for weight in weights]))
        return rows


def _stub_affect_payload() -> dict:
    return {"model": StubAffectRegressor(), "training_examples": 0}


def _stub_subtext_payload() -> dict:
    return {
        "features": StubFeatureExtractor(),
        "act_model": StubSignalClassifier(ACT_LABELS, 3),
        "emotion_model": StubSignalClassifier(EMOTION_LABELS, 5),
        "training_examples": 0,
        "act_confidence_threshold": 0.40,
        "emotion_confidence_threshold": 0.40,
    }


class _StubJoblib:
    """A ``joblib`` stand-in that returns fixed payloads per artifact path."""

    def __init__(self, payloads: dict[Path, dict]) -> None:
        self._payloads = payloads

    def load(self, path) -> dict:
        return self._payloads[Path(path)]


@pytest.fixture()
def stub_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the real loaders through stub fitted estimators."""
    payloads = {
        AFFECT_ARTIFACT: _stub_affect_payload(),
        SUBTEXT_ARTIFACT: _stub_subtext_payload(),
    }
    monkeypatch.setitem(sys.modules, "joblib", _StubJoblib(payloads))
    monkeypatch.setattr(
        TrainedIntent,
        "from_bundled",
        classmethod(lambda cls, *args, **kwargs: TrainedIntent(model=StubIntentClassifier())),
    )


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _intent_state(messages: list[dict]) -> object:
    return TrainedIntent(model=StubIntentClassifier()).infer(messages)


# --- classifier repeats ----------------------------------------------------


@pytest.mark.parametrize("messages", MESSAGES, ids=lambda row: row[-1]["content"][:24])
def test_repeated_intent_inference_is_byte_identical(messages: list[dict]) -> None:
    adapter = TrainedIntent(model=StubIntentClassifier())
    runs = [_canonical(asdict(adapter.infer(messages))) for _ in range(5)]
    assert len(set(runs)) == 1
    # The classifier also has to produce a real posterior, not an empty one.
    state = adapter.infer(messages)
    assert state.belief is not None
    assert state.risk_decision is not None


def test_independently_constructed_classifiers_agree() -> None:
    """Two fresh stub models must not diverge: no shared mutable state."""
    left = TrainedIntent(model=StubIntentClassifier())
    right = TrainedIntent(model=StubIntentClassifier())
    for messages in MESSAGES:
        assert _canonical(asdict(left.infer(messages))) == _canonical(
            asdict(right.infer(messages))
        )


def test_affect_regressor_predictions_stay_on_the_embobank_scale() -> None:
    adapter = TrainedAffect(model=StubAffectRegressor())
    for messages in MESSAGES:
        state = adapter.infer(messages, turn_index=len(messages))
        assert -1.0 <= state.valence <= 1.0
        assert 0.0 <= state.arousal <= 1.0
        for previous in state.history:
            assert -1.0 <= previous.valence <= 1.0
            assert 0.0 <= previous.arousal <= 1.0


# --- pipeline repeats ------------------------------------------------------


def _load_stub_pipeline() -> FullEQPipeline:
    return FullEQPipeline.load(AFFECT_ARTIFACT, SUBTEXT_ARTIFACT)


def test_repeated_steering_with_stub_artifacts_is_byte_identical(
    stub_artifacts: None,
) -> None:
    pipeline = _load_stub_pipeline()
    for messages in MESSAGES:
        runs = [_canonical(pipeline.steer_messages(messages)[1]) for _ in range(5)]
        assert len(set(runs)) == 1, messages[-1]["content"]


def test_two_independently_loaded_stub_pipelines_agree(stub_artifacts: None) -> None:
    left = _load_stub_pipeline()
    right = _load_stub_pipeline()
    for messages in MESSAGES:
        assert _canonical(left.steer_messages(messages)[1]) == _canonical(
            right.steer_messages(messages)[1]
        )


def test_steering_is_independent_of_the_order_cases_are_routed_in(
    stub_artifacts: None,
) -> None:
    pipeline = _load_stub_pipeline()
    forward = {_canonical(messages): pipeline.steer_messages(messages)[1] for messages in MESSAGES}
    reversed_map = {
        _canonical(messages): pipeline.steer_messages(messages)[1]
        for messages in reversed(MESSAGES)
    }
    assert reversed_map == forward


def test_the_full_learned_route_is_byte_identical(stub_artifacts: None) -> None:
    from eq_layer import cli

    runs = []
    for _ in range(3):
        runs.append(
            [
                _canonical(cli._full_route(messages, AFFECT_ARTIFACT, SUBTEXT_ARTIFACT))
                for messages in MESSAGES
            ]
        )
    assert runs[0] == runs[1] == runs[2]


def test_the_stub_artifacts_reach_the_learned_subtext_decisions(
    stub_artifacts: None,
) -> None:
    """Guard against a vacuous determinism test: the learned path must run.

    ``TrainedSubtext.infer`` reports which branch produced the label. If every
    stub artifact silently fell through to the structural fallback, the repeats
    above would only be pinning the heuristic, not the classifier.
    """
    pipeline = _load_stub_pipeline()
    levels = set()
    for messages in MESSAGES:
        _, metadata = pipeline.steer_messages(messages)
        levels.add(metadata["subtext_evidence_level"])
    assert "direct_learned" in levels


def test_learned_intent_posterior_is_a_full_distribution(stub_artifacts: None) -> None:
    pipeline = _load_stub_pipeline()
    for messages in MESSAGES:
        _, metadata = pipeline.steer_messages(messages)
        belief = metadata["intent_belief"]
        assert belief is not None
        assert set(belief) == set(INTENT_LABELS)
        assert abs(sum(belief.values()) - 1.0) < 1e-9
        assert metadata["intent_risk_decision"] is not None
        assert set(metadata["intent_risk_decision"]) >= {"action", "losses", "rationale"}
