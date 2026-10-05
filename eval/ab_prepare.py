"""Prepare a blinded A/B ballot from baseline and EQ-Layer responses.

Input JSONL rows:
{"id": "...", "context": [...], "baseline": "...", "eq": "..."}

Usage:
  python eval/ab_prepare.py pairs.jsonl ballot.jsonl key.json --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.ab_eval import load_jsonl, prepare_blinded, write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs")
    parser.add_argument("ballot")
    parser.add_argument("key")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pairs = load_jsonl(args.pairs)
    ballot, key = prepare_blinded(pairs, seed=args.seed)
    write_jsonl(args.ballot, ballot)

    with open(args.key, "w", encoding="utf-8") as fh:
        json.dump(key, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"prepared {len(ballot)} blinded cases")
    print("Keep the key hidden from raters until annotation is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
