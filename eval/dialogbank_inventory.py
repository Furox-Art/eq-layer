"""Inventory direct ISO 24617-2 higher-level labels in DialogBank English.

This is a data-availability audit. It does not train a classifier.

Usage: python eval/dialogbank_inventory.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.dialogbank_iso import (  # noqa: E402
    discover_diaml_urls,
    fetch_text,
    inventory,
    parse_diaml,
)


TARGETS = (
    "correction",
    "disagreement",
    "agreement",
    "selfCorrection",
    "correctMisspeaking",
)


def main() -> int:
    examples = []
    failures = []
    for url in discover_diaml_urls():
        try:
            examples.extend(parse_diaml(fetch_text(url), url))
        except Exception as exc:
            failures.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})

    report = inventory(examples)
    report["parse_failures"] = failures
    report["target_counts"] = {
        target: report["functions"].get(target, 0)
        for target in TARGETS
    }
    print(json.dumps(report, indent=2))

    if report["dialogues"] < 5 or report["examples"] < 100:
        return 1
    if len(failures) >= max(3, report["dialogues"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
