"""Train and evaluate direct disagreement detection on Coarse Discourse.

The split is conversation-grouped to avoid thread leakage. The decision
threshold is selected on dev only; test remains untouched.

Usage: python eval/coarse_disagreement_eval.py
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from eq_layer.coarse_discourse import (  # noqa: E402
    download_archive,
    load_examples,
    sha256_bytes,
)
from eq_layer.trained_disagreement import (  # noqa: E402
    COARSE_DISCOURSE_ARCHIVE_SHA256,
    TrainedDisagreement,
)


def split_grouped(examples: list) -> tuple[list, list, list]:
    groups = np.asarray([x.conversation_id for x in examples])
    indices = np.arange(len(examples))

    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=0)
    train_idx, hold_idx = next(first.split(indices, groups=groups))

    hold_groups = groups[hold_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=1)
    dev_rel, test_rel = next(
        second.split(np.arange(len(hold_idx)), groups=hold_groups)
    )

    dev_idx = hold_idx[dev_rel]
    test_idx = hold_idx[test_rel]

    return (
        [examples[int(i)] for i in train_idx],
        [examples[int(i)] for i in dev_idx],
        [examples[int(i)] for i in test_idx],
    )


def labels(rows: list) -> np.ndarray:
    return np.asarray(
        [1 if row.label == "disagreement" else 0 for row in rows],
        dtype=int,
    )


def probabilities(model: TrainedDisagreement, rows: list) -> np.ndarray:
    texts = [row.model_text for row in rows]
    return np.asarray(model.model.predict_proba(texts)[:, 1], dtype=float)


def choose_threshold(expected: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.50
    best_f1 = -1.0
    for threshold in np.linspace(0.30, 0.90, 61):
        predicted = (probs >= threshold).astype(int)
        score = float(f1_score(expected, predicted, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return round(best_threshold, 2), round(best_f1, 4)


def choose_high_precision_threshold(
    expected: np.ndarray,
    probs: np.ndarray,
    target_precision: float = 0.60,
) -> tuple[float, dict]:
    precisions, recalls, thresholds = precision_recall_curve(expected, probs)
    candidates = []
    for idx, threshold in enumerate(thresholds):
        precision = float(precisions[idx])
        recall = float(recalls[idx])
        if precision >= target_precision:
            candidates.append((recall, float(threshold), precision))

    if not candidates:
        threshold = float(np.nextafter(float(probs.max()), np.inf))
    else:
        _, threshold, _ = max(candidates)

    return threshold, metrics(expected, probs, threshold)


def metrics(expected: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    predicted = (probs >= threshold).astype(int)
    majority = np.zeros_like(expected)
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
        "majority_accuracy": round(float((majority == expected).mean()), 4),
        "positive_rate": round(float(expected.mean()), 4),
    }


def main() -> int:
    archive = download_archive()
    digest = sha256_bytes(archive)
    if digest != COARSE_DISCOURSE_ARCHIVE_SHA256:
        raise RuntimeError(
            "Coarse Discourse archive changed: "
            f"expected {COARSE_DISCOURSE_ARCHIVE_SHA256}, got {digest}"
        )

    examples, _ = load_examples(archive)
    train, dev, test = split_grouped(examples)

    model = TrainedDisagreement().fit(train)
    dev_y = labels(dev)
    dev_p = probabilities(model, dev)
    threshold, dev_best_f1 = choose_threshold(dev_y, dev_p)
    high_precision_threshold, dev_high_precision = choose_high_precision_threshold(
        dev_y,
        dev_p,
        target_precision=0.60,
    )

    test_y = labels(test)
    test_p = probabilities(model, test)

    report = {
        "benchmark": "Coarse-Discourse-direct-disagreement",
        "archive_sha256": digest,
        "split": "conversation-grouped 70/15/15",
        "train_examples": len(train),
        "dev_examples": len(dev),
        "test_examples": len(test),
        "train_conversations": len({x.conversation_id for x in train}),
        "dev_conversations": len({x.conversation_id for x in dev}),
        "test_conversations": len({x.conversation_id for x in test}),
        "train_positive": int(labels(train).sum()),
        "dev_positive": int(dev_y.sum()),
        "test_positive": int(test_y.sum()),
        "threshold_selected_on_dev": threshold,
        "dev_best_f1": dev_best_f1,
        "high_precision_threshold_selected_on_dev": high_precision_threshold,
        "dev": metrics(dev_y, dev_p, threshold),
        "test": metrics(test_y, test_p, threshold),
        "dev_high_precision": dev_high_precision,
        "test_high_precision": metrics(
            test_y,
            test_p,
            high_precision_threshold,
        ),
    }
    print(json.dumps(report, indent=2))

    values = []
    for split in ("dev", "test", "dev_high_precision", "test_high_precision"):
        values.extend(report[split].values())
    if not all(math.isfinite(float(v)) for v in values):
        return 1

    # Do not fail CI for a scientifically weak model. The executable/data
    # integrity checks above are hard failures; quality is reported verbatim.
    if report["test_positive"] < 100:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
