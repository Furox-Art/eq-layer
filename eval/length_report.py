"""Report the length gap between experiment arms.

The EQ arm runs about 3x shorter than the raw baseline, which is why EQ-vs-baseline
cannot separate the deterministic decision from the surface budget it implies.
This measures the gap per arm and reports the ratio between EQ and the
length-matched control, which is the comparison the design is for.

Reports words and characters because a token or character count alone can move
with the model's tokenizer rather than with the amount of content.

Usage:
  python eval/length_report.py pairs.jsonl --output length.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.ab_eval import load_jsonl  # noqa: E402

ARMS = ("baseline", "eq", "length_matched")


def describe(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 1),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def arm_lengths(row: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for arm in ARMS:
        text = row.get(arm)
        if not isinstance(text, str):
            continue
        out[arm] = {
            "characters": len(text),
            "words": len(text.split()),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs")
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = load_jsonl(args.pairs)
    per_case = [arm_lengths(row) for row in rows]
    present = [arm for arm in ARMS if any(arm in item for item in per_case)]

    totals = {
        unit: {arm: describe([item[arm][unit] for item in per_case if arm in item]) for arm in present}
        for unit in ("characters", "words")
    }

    ratios: dict[str, dict] = {}
    if "eq" in present:
        ratios["eq_vs_baseline"] = {
            unit: round(
                totals[unit]["eq"]["mean"] / totals[unit]["baseline"]["mean"], 3
            )
            for unit in ("characters", "words")
            if totals[unit]["baseline"]["mean"]
        }
    if "length_matched" in present:
        ratios["eq_vs_length_matched"] = {
            unit: round(
                totals[unit]["eq"]["mean"] / totals[unit]["length_matched"]["mean"], 3
            )
            for unit in ("characters", "words")
            if totals[unit]["length_matched"]["mean"]
        }
        ratios["length_matched_vs_baseline"] = {
            unit: round(
                totals[unit]["length_matched"]["mean"] / totals[unit]["baseline"]["mean"], 3
            )
            for unit in ("characters", "words")
            if totals[unit]["baseline"]["mean"]
        }

    missing = [row.get("id") for row, item in zip(rows, per_case) if "eq" not in item or "baseline" not in item]
    empty = [
        row.get("id")
        for row, item in zip(rows, per_case)
        if any(value == 0 for arm in item.values() for value in arm.values())
    ]

    report = {
        "audit": "arm-length-comparison",
        "n_cases": len(rows),
        "arms_present": present,
        "totals": totals,
        "ratios": ratios,
        "cases_missing_an_arm": missing,
        "cases_with_empty_arm": empty,
        "interpretation": (
            "If eq_vs_baseline is large while eq_vs_length_matched is near 1.0, the "
            "control arm worked: EQ is now being read against a same-length "
            "response instead of against the model's own verbosity."
        ),
        "claim_boundary": (
            "Length parity is a precondition for interpreting a preference test, not "
            "evidence of quality. No rating has been collected here."
        ),
    }

    text = json.dumps(report, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())