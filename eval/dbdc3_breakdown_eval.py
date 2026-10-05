"""Train/evaluate DBDC3 hard-breakdown detection on revised English data.

The official revised development dialogues provide training/calibration data.
The revised evaluation dialogues remain untouched until final testing.

Usage: python eval/dbdc3_breakdown_eval.py
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from eq_layer.dbdc3 import (  # noqa: E402
    DBDC3_ARCHIVE_SHA256,
    download_archive,
    parse_english_turns,
    sha256_bytes,
)
from eq_layer.trained_breakdown import TrainedBreakdown  # noqa: E402


def labels(rows: list) -> np.ndarray:
    return np.asarray([1 if row.hard_breakdown else 0 for row in rows], dtype=int)


def probabilities(model: TrainedBreakdown, rows: list) -> np.ndarray:
    texts = [row.model_text for row in rows]
    return np.asarray(model.model.predict_proba(texts)[:, 1], dtype=float)


def metrics(expected: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    predicted = (probs >= threshold).astype(int)
    return {
        "average_precision": round(float(average_precision_score(expected, probs)), 4),
        "roc_auc": round(float(roc_auc_score(expected, probs)), 4),
        "f1": round(float(f1_score(expected, predicted, zero_division=0)), 4),
        "precision": round(
            float(precision_score(expected, predicted, zero_division=0)),
            4,
        ),
        "recall": round(float(recall_score(expected, predicted, zero_division=0)), 4),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(expected, predicted)),
            4,
        ),
        "positive_rate": round(float(expected.mean()), 4),
    }


def choose_threshold(
    expected: np.ndarray,
    probs: np.ndarray,
    target_precision: float = 0.70,
) -> tuple[float, dict]:
    candidates = []
    for threshold in np.linspace(0.30, 0.99, 70):
        predicted = (probs >= threshold).astype(int)
        precision = float(precision_score(expected, predicted, zero_division=0))
        recall = float(recall_score(expected, predicted, zero_division=0))
        if precision >= target_precision:
            candidates.append((recall, float(threshold), precision))

    if not candidates:
        threshold = 0.99
    else:
        _, threshold, _ = max(candidates)

    threshold = round(threshold, 2)
    return threshold, metrics(expected, probs, threshold)


def main() -> int:
    archive = download_archive()
    digest = sha256_bytes(archive)
    if digest != DBDC3_ARCHIVE_SHA256:
        raise RuntimeError(
            f"DBDC3 archive changed: expected {DBDC3_ARCHIVE_SHA256}, got {digest}"
        )

    turns, meta = parse_english_turns(archive)
    dev = [row for row in turns if row.split == "dev"]
    test = [row for row in turns if row.split == "eval"]

    if not dev or not test:
        raise RuntimeError(f"Missing official revised split: dev={len(dev)} eval={len(test)}")

    groups = np.asarray([row.dialogue_id for row in dev])
    indices = np.arange(len(dev))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=0)
    train_idx, calibration_idx = next(splitter.split(indices, groups=groups))

    train = [dev[int(i)] for i in train_idx]
    calibration = [dev[int(i)] for i in calibration_idx]

    provisional = TrainedBreakdown().fit(train)
    calibration_y = labels(calibration)
    calibration_p = probabilities(provisional, calibration)
    threshold, calibration_metrics = choose_threshold(
        calibration_y,
        calibration_p,
        target_precision=0.70,
    )

    final_model = TrainedBreakdown(threshold=threshold).fit(dev)
    test_y = labels(test)
    test_p = probabilities(final_model, test)

    report = {
        "benchmark": "DBDC3-revised-English-hard-breakdown",
        "archive_sha256": digest,
        "source": meta["effective_source"],
        "official_dev_turns": len(dev),
        "official_eval_turns": len(test),
        "official_dev_dialogues": len({row.dialogue_id for row in dev}),
        "official_eval_dialogues": len({row.dialogue_id for row in test}),
        "dev_hard_breakdown": int(labels(dev).sum()),
        "eval_hard_breakdown": int(test_y.sum()),
        "calibration_dialogues": len({row.dialogue_id for row in calibration}),
        "threshold_selected_on_dev_calibration": threshold,
        "calibration": calibration_metrics,
        "test": metrics(test_y, test_p, threshold),
    }
    print(json.dumps(report, indent=2))

    values = list(report["calibration"].values()) + list(report["test"].values())
    if not all(math.isfinite(float(value)) for value in values):
        return 1
    if report["eval_hard_breakdown"] < 100:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
