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

CIRCULAR_STRATA = {"task_repair"}


def circular_ids(ballot: list[dict]) -> set[str]:
    """Case ids whose stratum membership is decided by the detector under test."""
    ids = set()
    for row in ballot:
        source = row.get("source") or {}
        if source.get("stratum") in CIRCULAR_STRATA:
            ids.add(str(row["id"]))
        elif row.get("stratum") in CIRCULAR_STRATA:
            ids.add(str(row["id"]))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rated_ballot")
    parser.add_argument("key")
    parser.add_argument(
        "--include-circular",
        action="store_true",
        help=(
            "Include strata whose membership is decided by the same phrase list the "
            "layer uses to detect them. Excluded by default because those cases "
            "inflate every dimension."
        ),
    )
    args = parser.parse_args()

    ballot = load_jsonl(args.rated_ballot)
    with open(args.key, encoding="utf-8") as fh:
        key = json.load(fh)

    exclude = set() if args.include_circular else circular_ids(ballot)

    report = score_blinded(ballot, key, exclude_ids=exclude)
    report["exclusions_applied"] = {
        "circular_strata_excluded": not args.include_circular,
        "circular_strata": sorted(CIRCULAR_STRATA),
        "n_excluded": len(exclude),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
