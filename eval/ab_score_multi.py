"""Score multiple completed blinded A/B ballots.

Usage:
  python eval/ab_score_multi.py key.json rater1.jsonl rater2.jsonl rater3.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.ab_eval import load_jsonl, score_multiple_blinded  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("key")
    parser.add_argument("rated_ballots", nargs="+")
    args = parser.parse_args()

    if len(args.rated_ballots) < 2:
        parser.error("provide at least two rated ballots")

    with open(args.key, encoding="utf-8") as fh:
        key = json.load(fh)

    ballots = {
        Path(path).stem: load_jsonl(path)
        for path in args.rated_ballots
    }
    report = score_multiple_blinded(ballots, key)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
