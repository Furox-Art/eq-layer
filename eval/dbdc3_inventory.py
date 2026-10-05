"""Audit English DBDC3 breakdown labels.

Usage: python eval/dbdc3_inventory.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.dbdc3 import (  # noqa: E402
    download_archive,
    inventory,
    parse_english_turns,
    sha256_bytes,
)


def main() -> int:
    data = download_archive()
    turns, meta = parse_english_turns(data)
    report = inventory(turns)
    report["archive_sha256"] = sha256_bytes(data)
    report.update(meta)
    print(json.dumps(report, indent=2))

    if report["system_turns"] < 1000:
        return 1
    if report["hard_breakdown_turns"] < 100:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
