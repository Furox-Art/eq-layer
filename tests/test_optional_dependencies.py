"""Regression tests for the optional-dependency boundary (audit M-2/M-7).

A missing or unusable ``ml`` extra must surface as one typed, actionable error
that names the package and the install command, never as a bare ``ImportError``
traceback at request time. The tests run without scikit-learn installed: the
import is made to fail by putting ``None`` into ``sys.modules``, which is how
CPython reports a halted import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from eq_layer import (
    OptionalDependencyError,
    cli,
    trained_affect,
    trained_breakdown,
    trained_disagreement,
    trained_intent,
    trained_subtext,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_COMMAND = 'pip install "eq-layer[ml]"'

# Every scikit-learn builder in the package.
ML_BUILDERS = (
    ("TrainedIntent", trained_intent.build_pipeline),
    ("TrainedAffect", trained_affect.build_affect_pipeline),
    ("TrainedSubtext", trained_subtext.build_feature_extractor),
    ("TrainedSubtext", trained_subtext.build_classifier),
    ("TrainedBreakdown", trained_breakdown.build_breakdown_pipeline),
    ("TrainedDisagreement", trained_disagreement.build_disagreement_pipeline),
)

# Every artifact loader that needs joblib, with the purpose it must name.
JOBLIB_LOADERS = (
    ("Loading a trained intent model", trained_intent.TrainedIntent),
    ("Loading a trained affect model", trained_affect.TrainedAffect),
    ("Loading a trained subtext model", trained_subtext.TrainedSubtext),
    ("Loading breakdown model", trained_breakdown.TrainedBreakdown),
    ("Loading disagreement model", trained_disagreement.TrainedDisagreement),
)

# Every artifact saver, with a fitted-enough instance to pass its own guard.
JOBLIB_SAVERS = (
    ("Saving a trained intent model", trained_intent.TrainedIntent, {"model": object()}),
    ("Saving a trained affect model", trained_affect.TrainedAffect, {"model": object()}),
    (
        "Saving a trained subtext model",
        trained_subtext.TrainedSubtext,
        {"features": object(), "act_model": object(), "emotion_model": object()},
    ),
    ("Saving breakdown model", trained_breakdown.TrainedBreakdown, {"model": object()}),
    ("Saving disagreement model", trained_disagreement.TrainedDisagreement, {"model": object()}),
)


@pytest.fixture
def sklearn_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sklearn", None)


@pytest.fixture
def joblib_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "joblib", None)


@pytest.mark.parametrize(("purpose", "builder"), ML_BUILDERS)
def test_ml_builders_map_the_missing_extra_to_a_typed_error(
    purpose: str,
    builder: object,
    sklearn_missing: None,
) -> None:
    with pytest.raises(OptionalDependencyError) as raised:
        builder()
    message = str(raised.value)
    assert purpose in message
    assert "scikit-learn" in message
    assert INSTALL_COMMAND in message
    # The typed error is still an ImportError, so existing fallbacks keep working.
    assert isinstance(raised.value, ImportError)


def test_typed_error_carries_the_install_command(sklearn_missing: None) -> None:
    with pytest.raises(OptionalDependencyError) as raised:
        trained_intent.build_pipeline()
    assert raised.value.install_command == INSTALL_COMMAND
    assert raised.value.package == "scikit-learn"


@pytest.mark.parametrize(("purpose", "adapter"), JOBLIB_LOADERS)
def test_artifact_loaders_map_the_missing_extra_to_a_typed_error(
    purpose: str,
    adapter: object,
    joblib_missing: None,
) -> None:
    with pytest.raises(OptionalDependencyError) as raised:
        adapter.load("artifacts/model.joblib")
    assert purpose in str(raised.value)
    assert "joblib" in str(raised.value)
    assert INSTALL_COMMAND in str(raised.value)


@pytest.mark.parametrize(("purpose", "adapter", "fitted"), JOBLIB_SAVERS)
def test_artifact_savers_map_the_missing_extra_to_a_typed_error(
    purpose: str,
    adapter: type,
    fitted: dict,
    joblib_missing: None,
    tmp_path: Path,
) -> None:
    with pytest.raises(OptionalDependencyError) as raised:
        adapter(**fitted).save(tmp_path / "model.joblib")
    assert purpose in str(raised.value)
    assert INSTALL_COMMAND in str(raised.value)


def test_cli_auto_mode_still_degrades_to_the_heuristic_backend(
    sklearn_missing: None,
) -> None:
    adapter, backend = cli._intent_adapter("auto")
    assert backend == "heuristic-fallback-no-ml-extra"
    assert adapter.infer([{"role": "user", "content": "hello"}]).kind


def test_cli_learned_mode_exits_cleanly_with_the_install_command(
    sklearn_missing: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trained_intent.clear_bundled_cache()
    payload = [{"role": "user", "content": "devam et"}]
    exit_code = cli.main(
        ["route", "--intent-mode", "learned", "--messages-json", json.dumps(payload)]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert INSTALL_COMMAND in captured.err
    assert "scikit-learn" in captured.err
    # The error goes to stderr only; stdout stays valid JSON-free, not a usage dump.
    assert "usage:" not in captured.err


def test_cli_doctor_reports_the_missing_extra(
    sklearn_missing: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["doctor"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ml_extra_available"] is False
    assert INSTALL_COMMAND in payload["ml_extra_hint"]


def test_every_optional_import_site_in_the_package_is_mapped() -> None:
    """Guard the fix itself: no bare ``raise ImportError`` may come back."""
    for source in REPO_ROOT.glob("eq_layer/*.py"):
        text = source.read_text(encoding="utf-8")
        assert "raise ImportError(" not in text, f"{source.name} still raises a bare ImportError"
