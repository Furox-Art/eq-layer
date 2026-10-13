"""Regression tests for the bundled intent cache (audit M-5).

``TrainedIntent.from_bundled()`` used to re-fit its classifier on every call,
so a request path that loads the bundled adapter once per request paid the full
fit each time. The fit is now memoised per process on
(artifact path, artifact mtime, thresholds); these tests count fits and check
that an edited artifact invalidates the memo.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eq_layer.trained_intent import (
    TrainedIntent,
    bundled_records_path,
    clear_bundled_cache,
    load_jsonl_records,
)


@pytest.fixture(autouse=True)
def clean_cache() -> None:
    clear_bundled_cache()
    yield
    clear_bundled_cache()


@pytest.fixture
def counting_fit(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the real fit (scikit-learn) with a counting stub."""
    calls: list[int] = []

    def _fit(self: TrainedIntent, records) -> TrainedIntent:
        rows = list(records)
        calls.append(len(rows))
        self.model = object()
        self.training_examples = len(rows)
        return self

    monkeypatch.setattr(TrainedIntent, "fit", _fit)
    return calls


def test_bundled_records_path_resolves_the_packaged_corpus() -> None:
    artifact = bundled_records_path()
    assert artifact.name == "intent_train.jsonl"
    assert len(load_jsonl_records(artifact)) > 1


def test_repeated_from_bundled_fits_once(counting_fit: list[int]) -> None:
    adapters = [TrainedIntent.from_bundled() for _ in range(5)]

    assert len(counting_fit) == 1
    # The cached instance is shared, so a repeated call is O(1).
    assert all(adapter is adapters[0] for adapter in adapters)
    assert adapters[0].model is not None


def test_thresholds_are_part_of_the_cache_key(counting_fit: list[int]) -> None:
    first = TrainedIntent.from_bundled(confidence_threshold=0.40, margin_threshold=0.08)
    second = TrainedIntent.from_bundled(confidence_threshold=0.55, margin_threshold=0.08)

    assert len(counting_fit) == 2
    assert first is not second
    assert (first.confidence_threshold, second.confidence_threshold) == (0.40, 0.55)


def test_edited_artifact_invalidates_the_cache(
    counting_fit: list[int],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "intent_train.jsonl"
    artifact.write_text(
        '{"text": "run the build", "label": "action_request"}\n'
        '{"text": "is it done", "label": "status_check"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "eq_layer.trained_intent.bundled_records_path",
        lambda: artifact,
    )
    monkeypatch.setattr(
        "eq_layer.trained_intent.bundled_training_records",
        lambda: load_jsonl_records(artifact),
    )

    assert TrainedIntent.from_bundled().training_examples == 2
    assert len(counting_fit) == 1

    # Rewriting the corpus changes its mtime, so the memo must be refilled.
    future = artifact.stat().st_mtime + 60
    os.utime(artifact, (future, future))
    assert TrainedIntent.from_bundled().training_examples == 2
    assert len(counting_fit) == 2


def test_pipeline_load_shares_one_bundled_fit(counting_fit: list[int]) -> None:
    from eq_layer.response_experiment import FullEQPipeline

    for _ in range(3):
        pipeline = FullEQPipeline(
            tracker=object(),  # type: ignore[arg-type]
            intent=TrainedIntent.from_bundled(),
            selector=object(),  # type: ignore[arg-type]
        )
        assert pipeline.intent is not None

    assert len(counting_fit) == 1
