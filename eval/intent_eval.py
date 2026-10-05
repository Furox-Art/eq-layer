"""Evaluate trained intent tracking on a fixed synthetic hold-out set.

This is a development benchmark, not an external or human-labelled standard.
It reports both a fixed hold-out score and five-fold cross-validation on the
bundled training corpus to make overfitting harder to hide.

Usage: python eval/intent_eval.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from eq_layer.intent import HeuristicIntent  # noqa: E402
from eq_layer.trained_intent import (  # noqa: E402
    TrainedIntent,
    build_pipeline,
    bundled_training_records,
    load_jsonl_records,
)

HOLDOUT = os.path.join(os.path.dirname(__file__), "intent_holdout.jsonl")


def predict(adapter, records: list[dict]) -> list[str]:
    return [
        adapter.infer([{"role": "user", "content": row["text"]}]).kind
        for row in records
    ]


def metrics(expected: list[str], predicted: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(expected, predicted), 4),
        "macro_f1": round(f1_score(expected, predicted, average="macro"), 4),
    }


def main() -> int:
    holdout = load_jsonl_records(HOLDOUT)
    expected = [row["label"] for row in holdout]

    heuristic = HeuristicIntent()
    trained = TrainedIntent.from_bundled()

    heuristic_scores = metrics(expected, predict(heuristic, holdout))
    trained_scores = metrics(expected, predict(trained, holdout))

    train = bundled_training_records()
    texts = [row["text"] for row in train]
    labels = [row["label"] for row in train]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(
        build_pipeline(),
        texts,
        labels,
        cv=cv,
        scoring="accuracy",
    )

    report = {
        "benchmark": "synthetic-development-only",
        "training_examples": len(train),
        "holdout_examples": len(holdout),
        "heuristic": heuristic_scores,
        "trained": trained_scores,
        "cv5_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv5_accuracy_min": round(float(cv_scores.min()), 4),
        "confidence_threshold": trained.confidence_threshold,
    }
    print(json.dumps(report, indent=2))

    # Guard against silent regressions without pretending this small synthetic
    # benchmark establishes real-world EQ or intent-understanding quality.
    if trained_scores["accuracy"] < 0.85:
        return 1
    if trained_scores["accuracy"] <= heuristic_scores["accuracy"]:
        return 1
    if report["cv5_accuracy_mean"] < 0.70:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
