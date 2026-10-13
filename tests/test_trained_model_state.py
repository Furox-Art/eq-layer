"""Regression tests for trained-model state validation (audit H-2).

A malformed artifact (missing key, empty payload, wrong field type) used to
escape as a raw ``KeyError``/``TypeError`` from inside the dataclass
constructor. Every loader in ``eq_layer/trained_*.py`` now validates the
payload schema first and raises a typed ``ModelStateError`` naming the offending
field.

No real artifacts are needed: ``joblib`` is replaced with a stub whose
``load`` returns the payload under test, so these tests run without
scikit-learn and without any binary fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from eq_layer import (
    ModelStateError,
    trained_affect,
    trained_breakdown,
    trained_disagreement,
    trained_intent,
    trained_subtext,
)

ARTIFACT = Path("artifacts/model.joblib")

# Every persisted loader, with a payload that must load cleanly.
LOADERS = (
    trained_affect.TrainedAffect,
    trained_intent.TrainedIntent,
    trained_subtext.TrainedSubtext,
    trained_breakdown.TrainedBreakdown,
    trained_disagreement.TrainedDisagreement,
)

PAYLOADS = {
    trained_affect.TrainedAffect: {
        "model": object(),
        "training_examples": 12,
        "training_source": "EmoBank",
    },
    trained_intent.TrainedIntent: {
        "model": object(),
        "confidence_threshold": 0.4,
        "margin_threshold": 0.08,
        "training_examples": 12,
    },
    trained_subtext.TrainedSubtext: {
        "features": object(),
        "act_model": object(),
        "emotion_model": object(),
        "training_examples": 12,
    },
    trained_breakdown.TrainedBreakdown: {
        "model": object(),
        "training_examples": 12,
        "threshold": 0.75,
    },
    trained_disagreement.TrainedDisagreement: {
        "model": object(),
        "training_examples": 12,
        "threshold": 0.7,
    },
}


@pytest.fixture
def joblib_stub(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """A stand-in for joblib that hands back a caller-supplied payload."""
    module = ModuleType("joblib")
    monkeypatch.setitem(sys.modules, "joblib", module)
    return module


@pytest.mark.parametrize("adapter", LOADERS, ids=lambda cls: cls.__name__)
def test_well_formed_state_still_loads(
    adapter: type,
    joblib_stub: ModuleType,
) -> None:
    payload = dict(PAYLOADS[adapter])
    joblib_stub.load = lambda _path: payload
    loaded = adapter.load(ARTIFACT)
    assert type(loaded) is adapter


@pytest.mark.parametrize("adapter", LOADERS, ids=lambda cls: cls.__name__)
def test_empty_payload_is_a_typed_error(
    adapter: type,
    joblib_stub: ModuleType,
) -> None:
    joblib_stub.load = lambda _path: {}
    with pytest.raises(ModelStateError) as raised:
        adapter.load(ARTIFACT)
    assert raised.value.field in PAYLOADS[adapter]
    assert "missing required key" in str(raised.value)


@pytest.mark.parametrize("adapter", LOADERS, ids=lambda cls: cls.__name__)
def test_missing_required_field_is_a_typed_error(
    adapter: type,
    joblib_stub: ModuleType,
) -> None:
    payload = dict(PAYLOADS[adapter])
    for required in ("model", "features", "confidence_threshold"):
        payload.pop(required, None)
    joblib_stub.load = lambda _path: payload
    with pytest.raises(ModelStateError) as raised:
        adapter.load(ARTIFACT)
    assert raised.value.field in {"model", "features", "confidence_threshold"}
    assert raised.value.model in {cls.__name__ for cls in LOADERS}


def test_subtext_reports_the_offending_features_field(joblib_stub: ModuleType) -> None:
    payload = {
        "act_model": object(),
        "emotion_model": object(),
        "training_examples": 12,
    }
    joblib_stub.load = lambda _path: payload
    with pytest.raises(ModelStateError) as raised:
        trained_subtext.TrainedSubtext.load(ARTIFACT)
    assert raised.value.field == "features"
    assert "TrainedSubtext artifact" in str(raised.value)
    assert str(ARTIFACT) in str(raised.value)


def test_wrong_field_type_is_a_typed_error(joblib_stub: ModuleType) -> None:
    # A string where a threshold number belongs: before the schema check this
    # was silently coerced by float(), so the artifact "loaded".
    joblib_stub.load = lambda _path: {
        "model": object(),
        "confidence_threshold": "0.4",
        "margin_threshold": 0.08,
    }
    with pytest.raises(ModelStateError) as raised:
        trained_intent.TrainedIntent.load(ARTIFACT)
    assert raised.value.field == "confidence_threshold"
    assert "expected int or float" in str(raised.value)
    assert "got str" in str(raised.value)


def test_none_where_a_fitted_model_belongs_is_a_typed_error(
    joblib_stub: ModuleType,
) -> None:
    joblib_stub.load = lambda _path: {
        "model": None,
        "confidence_threshold": 0.4,
    }
    with pytest.raises(ModelStateError) as raised:
        trained_intent.TrainedIntent.load(ARTIFACT)
    assert raised.value.field == "model"
    assert "must not be None" in str(raised.value)


def test_non_mapping_payload_is_a_typed_error(joblib_stub: ModuleType) -> None:
    joblib_stub.load = lambda _path: ["model", "confidence_threshold"]
    with pytest.raises(ModelStateError) as raised:
        trained_affect.TrainedAffect.load(ARTIFACT)
    assert raised.value.field == "payload"
    assert "expected a mapping" in str(raised.value)


def test_non_numeric_threshold_on_an_experimental_loader_is_a_typed_error(
    joblib_stub: ModuleType,
) -> None:
    joblib_stub.load = lambda _path: {
        "model": object(),
        "threshold": "high",
        "training_examples": 1,
    }
    with pytest.raises(ModelStateError) as raised:
        trained_breakdown.TrainedBreakdown.load(ARTIFACT)
    assert raised.value.field == "threshold"


def test_valid_state_of_every_experimental_loader_round_trips(
    joblib_stub: ModuleType,
) -> None:
    joblib_stub.load = lambda _path: {
        "model": object(),
        "training_examples": 500,
        "positive_examples": 300,
        "threshold": 0.7,
    }
    loaded = trained_disagreement.TrainedDisagreement.load(ARTIFACT)
    assert loaded.training_examples == 500
    assert loaded.positive_examples == 300
    assert loaded.threshold == 0.7
