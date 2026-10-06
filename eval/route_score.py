"""Score the deterministic routing adjudication ballot.

This is not a pairwise preference problem. Each row asks a human to name the
task move the assistant should make, and the question is whether the layer's
decision matches. So there are no wins and losses, only agreement, and the
statistic is a binomial proportion with a Wilson interval rather than a sign
test.

Two exclusions are applied by default, because both inflate the number:

- **Circular strata.** task_repair is selected into the held-out set by
  REPAIR_PHRASES and detected by CORRECTION_MARKERS, six of seven phrases
  shared. Agreement there measures recovery of the selection criterion.
- **Unknown-intent cases.** 53 of 120 cases route on intent_kind=unknown and
  fall through to the clarify/respond defaults. Pooling them hides how much of
  the agreement is a fallback guessing convention rather than a decision.

Both are reported separately so nothing is hidden, and both can be re-included
explicitly for inspection.

Usage:
  python eval/route_score.py rated_route_ballot.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.ab_eval import load_jsonl  # noqa: E402

CIRCULAR_STRATA = {"task_repair"}


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1.0 + (z * z) / total
    centre = (p + (z * z) / (2.0 * total)) / denom
    margin = (
        z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total) / denom
    )
    return round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)


def agreement(successes: int, total: int) -> dict:
    interval = wilson(successes, total)
    return {
        "agreed": successes,
        "n": total,
        "agreement_rate": round(successes / total, 4) if total else None,
        "wilson_95_low": interval[0] if interval else None,
        "wilson_95_high": interval[1] if interval else None,
    }


def collect(rated_ballot: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split rated rows from unrated ones, validating as we go."""
    rated: list[dict] = []
    problems: list[dict] = []

    for row in rated_ballot:
        case_id = str(row.get("id"))
        decision = row.get("eq_layer_decision") or {}
        adjudication = row.get("adjudication") or {}
        allowed = adjudication.get("acceptable_task_moves") or []
        verdict = str(adjudication.get("acceptable_task_move", "")).strip()

        if not verdict:
            problems.append({"id": case_id, "issue": "unrated"})
            continue
        if allowed and verdict not in allowed:
            raise ValueError(
                f"{case_id}: adjudicated move {verdict!r} not in allowed set {allowed}"
            )

        predicted = decision.get("task_move")
        if not predicted:
            raise ValueError(f"{case_id}: ballot row has no eq_layer_decision.task_move")

        stratum = row.get("stratum")
        circular = (
            stratum in CIRCULAR_STRATA
            if stratum
            else bool(adjudication.get("circular_stratum"))
        )

        rated.append(
            {
                "id": case_id,
                "stratum": stratum,
                "predicted": predicted,
                "adjudicated": verdict,
                "agreed": predicted == verdict,
                "circular": circular,
                "intent_known": (decision.get("intent_kind") or "unknown") != "unknown",
            }
        )
    return rated, problems


def report(rated: list[dict], problems: list[dict], *, include_circular: bool, include_unknown_intent: bool) -> dict:
    def subset(rows: list[dict]) -> list[dict]:
        out = rows
        if not include_circular:
            out = [r for r in out if not r["circular"]]
        if not include_unknown_intent:
            out = [r for r in out if r["intent_known"]]
        return out

    headline = subset(rated)
    by_stratum = {
        name: agreement(sum(r["agreed"] for r in rows), len(rows))
        for name, rows in sorted(
            ((name, [r for r in rated if r.get("stratum") == name])
             for name in {r.get("stratum") for r in rated}),
            key=lambda kv: kv[0] or "",
        )
    }

    disagreements = [
        {"id": r["id"], "stratum": r["stratum"], "predicted": r["predicted"], "adjudicated": r["adjudicated"]}
        for r in headline
        if not r["agreed"]
    ]

    return {
        "design": "blind-single-label-routing-adjudication",
        "unit": "case-level task-move agreement",
        "n_rows": len(rated) + len(problems),
        "n_rated": len(rated),
        "n_unrated": len(problems),
        "exclusions_applied": {
            "circular_strata_excluded": not include_circular,
            "unknown_intent_excluded": not include_unknown_intent,
            "circular_strata": sorted(CIRCULAR_STRATA),
        },
        "headline": agreement(sum(r["agreed"] for r in headline), len(headline)),
        "all_rated": agreement(sum(r["agreed"] for r in rated), len(rated)),
        "by_stratum": by_stratum,
        "predicted_distribution": dict(
            sorted(Counter(r["predicted"] for r in rated).items())
        ),
        "adjudicated_distribution": dict(
            sorted(Counter(r["adjudicated"] for r in rated).items())
        ),
        "headline_disagreements": disagreements,
        "unrated_ids": [p["id"] for p in problems],
        "claim_boundary": (
            "Agreement between the deterministic task_move and a single human "
            "adjudication. It is not a response-quality result, and with one rater "
            "per case it carries no inter-rater reliability."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rated_ballot")
    parser.add_argument(
        "--include-circular",
        action="store_true",
        help="Include strata whose membership matches the detector under test.",
    )
    parser.add_argument(
        "--include-unknown-intent",
        action="store_true",
        help="Include cases where intent_kind is unknown.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    rated, problems = collect(load_jsonl(args.rated_ballot))
    payload = report(
        rated,
        problems,
        include_circular=args.include_circular,
        include_unknown_intent=args.include_unknown_intent,
    )
    text = json.dumps(payload, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())