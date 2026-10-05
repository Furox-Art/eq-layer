"""Train/evaluate learned dialogue signals on pinned XDailyDialog English splits.

XDailyDialog directly supervises dialogue act and basic emotion. The derived
EQ-Layer subtext labels are not treated as upstream ground truth.

Usage: python eval/xdailydialog_subtext_eval.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from eq_layer.trained_subtext import (  # noqa: E402
    XDAILYDIALOG_COMMIT,
    XDAILYDIALOG_URLS,
    TrainedSubtext,
    load_xdailydialog,
)


def download_split(split: str, directory: str) -> str:
    path = os.path.join(directory, f"{split}.txt")
    urllib.request.urlretrieve(XDAILYDIALOG_URLS[split], path)
    return path


def majority_accuracy(labels: list[str]) -> float:
    count = Counter(labels)
    return max(count.values()) / len(labels)


def evaluate(model: TrainedSubtext, rows: list[dict]) -> dict:
    texts = [row["text"] for row in rows]
    expected_act = [row["act"] for row in rows]
    expected_emotion = [row["emotion"] for row in rows]

    matrix = model.features.transform(texts)
    predicted_act = model.act_model.predict(matrix)
    predicted_emotion = model.emotion_model.predict(matrix)

    return {
        "act": {
            "accuracy": round(float(accuracy_score(expected_act, predicted_act)), 4),
            "macro_f1": round(
                float(f1_score(expected_act, predicted_act, average="macro")),
                4,
            ),
            "majority_accuracy": round(majority_accuracy(expected_act), 4),
        },
        "emotion": {
            "accuracy": round(float(accuracy_score(expected_emotion, predicted_emotion)), 4),
            "macro_f1": round(
                float(f1_score(expected_emotion, predicted_emotion, average="macro")),
                4,
            ),
            "majority_accuracy": round(majority_accuracy(expected_emotion), 4),
        },
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="eq-layer-xdailydialog-") as tmp:
        paths = {split: download_split(split, tmp) for split in ("train", "dev", "test")}
        train = load_xdailydialog(paths["train"])
        dev = load_xdailydialog(paths["dev"])
        test = load_xdailydialog(paths["test"])

        model = TrainedSubtext().fit(train)
        report = {
            "benchmark": "XDailyDialog-English-official-files",
            "upstream_commit": XDAILYDIALOG_COMMIT,
            "scope": (
                "dialogue-act and basic-emotion supervision only; EQ-Layer subtext "
                "labels are derived downstream"
            ),
            "train_examples": len(train),
            "dev_examples": len(dev),
            "test_examples": len(test),
            "dev": evaluate(model, dev),
            "test": evaluate(model, test),
        }
        print(json.dumps(report, indent=2))

        values = []
        for split in ("dev", "test"):
            for task in ("act", "emotion"):
                values.extend(report[split][task].values())

        if not all(math.isfinite(float(value)) for value in values):
            return 1

        # Parser/model sanity only. Weak scientific results must remain visible
        # rather than being hidden by a hand-tuned pass threshold.
        if len(train) < 1000 or len(dev) < 100 or len(test) < 100:
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
