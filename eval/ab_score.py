"""Score a completed blinded A/B ballot.

Each ballot row must have ratings for:
intent_fidelity, appropriateness, actionability, non_patronizing, overall.

Each rating is A, B, or tie.

Usage:
  python eval/ab_score.py rated_ballot.jsonl key.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.ab_eval import load_jsonl, score_blinded  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rated_ballot")
    parser.add_argument("key")
    args = parser.parse_args()

    ballot = load_jsonl(args.rated_ballot)
    with open(args.key, encoding="utf-8") as fh:
        key = json.load(fh)

    report = score_blinded(ballot, key)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
