"""Inventory direct ISO 24617-2 higher-level labels in DialogBank English.

This is a data-availability audit. It does not train a classifier.

Usage: python eval/dialogbank_inventory.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.dialogbank_iso import inventory, load_dialogbank_english  # noqa: E402


TARGETS = (
    "correction",
    "disagreement",
    "agreement",
    "selfCorrection",
    "correctMisspeaking",
)


def main() -> int:
    examples = load_dialogbank_english()
    report = inventory(examples)
    report["target_counts"] = {
        target: report["functions"].get(target, 0)
        for target in TARGETS
    }
    print(json.dumps(report, indent=2))

    if report["dialogues"] < 5 or report["examples"] < 100:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
