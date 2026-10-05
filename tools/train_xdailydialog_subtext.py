"""Train learned dialogue signals from pinned XDailyDialog English data.

Usage:
    python tools/train_xdailydialog_subtext.py --output artifacts/xdailydialog_subtext.joblib

The source corpus is downloaded at runtime and is not vendored.
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

from eq_layer.trained_subtext import (
    XDAILYDIALOG_COMMIT,
    XDAILYDIALOG_URLS,
    TrainedSubtext,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="eq-layer-xdailydialog-") as tmp:
        train_path = Path(tmp) / "train.txt"
        urllib.request.urlretrieve(XDAILYDIALOG_URLS["train"], train_path)
        model = TrainedSubtext.from_xdailydialog(train_path)
        model.save(output)

    print(f"saved {output} ({model.training_examples} training examples)")
    print(f"source=XDailyDialog@{XDAILYDIALOG_COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
