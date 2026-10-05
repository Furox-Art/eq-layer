"""Audit the direct disagreement signal in the CC-BY Coarse Discourse corpus.

Usage: python eval/coarse_discourse_inventory.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.coarse_discourse import (  # noqa: E402
    download_archive,
    inventory,
    load_examples,
    sha256_bytes,
)


def main() -> int:
    data = download_archive()
    examples, names = load_examples(data)
    report = inventory(examples)
    report["archive_sha256"] = sha256_bytes(data)
    report["zip_members"] = len(names)
    report["disagreement_examples"] = report["labels"].get("disagreement", 0)
    print(json.dumps(report, indent=2))

    if report["examples"] < 50000:
        return 1
    if report["disagreement_examples"] < 1000:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
