"""Download pinned EmoBank, train EQ-Layer affect regression, save a local model.

Usage:
    python tools/train_emobank_affect.py --output artifacts/emobank_affect.joblib

The dataset itself is not copied into the repository.
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

from eq_layer.trained_affect import EMOBANK_URL, TrainedAffect


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="eq-layer-emobank-") as tmp:
        csv_path = Path(tmp) / "emobank.csv"
        urllib.request.urlretrieve(EMOBANK_URL, csv_path)
        model = TrainedAffect.from_emobank(csv_path, split="train")
        model.save(output)

    print(f"saved {output} ({model.training_examples} training examples)")
    print(f"source={model.training_source} license={model.training_license}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
