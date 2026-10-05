"""Train and evaluate EQ-Layer's affect regressor on official EmoBank splits.

The dataset is downloaded from a pinned upstream commit and is not vendored in
this repository. EmoBank is CC-BY-SA-4.0; see THIRD_PARTY_DATA.md.

Usage: python eval/emobank_affect_eval.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402

from eq_layer.trained_affect import (  # noqa: E402
    EMOBANK_COMMIT,
    EMOBANK_LICENSE,
    EMOBANK_URL,
    TrainedAffect,
    load_emobank,
)


def evaluate(model: TrainedAffect, records: list[dict], train_records: list[dict]) -> dict:
    texts = [row["text"] for row in records]
    expected = np.asarray([[row["V"], row["A"], row["D"]] for row in records], dtype=float)
    predicted = np.asarray(model.model.predict(texts), dtype=float)

    train_targets = np.asarray(
        [[row["V"], row["A"], row["D"]] for row in train_records],
        dtype=float,
    )
    baseline = np.tile(train_targets.mean(axis=0), (len(records), 1))

    dimensions = ("V", "A", "D")
    out: dict[str, dict] = {}
    for idx, dim in enumerate(dimensions):
        rho = float(spearmanr(expected[:, idx], predicted[:, idx]).statistic)
        out[dim] = {
            "mae": round(float(mean_absolute_error(expected[:, idx], predicted[:, idx])), 4),
            "mean_baseline_mae": round(
                float(mean_absolute_error(expected[:, idx], baseline[:, idx])),
                4,
            ),
            "spearman_rho": round(rho, 4),
        }
    return out


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="eq-layer-emobank-") as tmp:
        path = os.path.join(tmp, "emobank.csv")
        urllib.request.urlretrieve(EMOBANK_URL, path)

        train = load_emobank(path, split="train")
        dev = load_emobank(path, split="dev")
        test = load_emobank(path, split="test")

        if not train or not dev or not test:
            raise RuntimeError(
                f"Official split missing: train={len(train)} dev={len(dev)} test={len(test)}"
            )

        model = TrainedAffect().fit(train)
        report = {
            "benchmark": "EmoBank-official-split",
            "upstream_commit": EMOBANK_COMMIT,
            "upstream_license": EMOBANK_LICENSE,
            "train_examples": len(train),
            "dev_examples": len(dev),
            "test_examples": len(test),
            "dev": evaluate(model, dev, train),
            "test": evaluate(model, test, train),
        }
        print(json.dumps(report, indent=2))

        # Scientific result policy: do not turn weak or negative results into a
        # CI failure. CI verifies data provenance, schema, numerical finiteness,
        # and executable integration. Quality metrics are reported verbatim.
        values = []
        for split in ("dev", "test"):
            for dim in ("V", "A", "D"):
                values.extend(report[split][dim].values())

        if not all(math.isfinite(float(value)) for value in values):
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
