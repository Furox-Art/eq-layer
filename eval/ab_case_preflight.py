"""Preflight held-out response-level A/B cases before generation.

Usage:
  python eval/ab_case_preflight.py heldout.jsonl
  python eval/ab_case_preflight.py heldout.jsonl --final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.response_experiment import validate_experiment_cases  # noqa: E402


def load(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()

    report = validate_experiment_cases(load(args.cases), final=args.final)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
