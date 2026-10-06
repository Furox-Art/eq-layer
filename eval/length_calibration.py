"""Choose the length-matched control profile by matching arm length.

Selection is on length only. Ranking profiles against the EQ arm's mean and
median word count and picking the smallest relative gap is a nuisance-variable
calibration; ranking them against which response a rater preferred would fit the
control to the outcome it exists to be compared against.

Both mean and median are used because the EQ arm is short and the distributions
are right-skewed, and the two can disagree. A profile that matches the mean but
not the median has not actually matched.

Usage:
  python eval/length_calibration.py pairs.jsonl --output calibration.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.ab_eval import load_jsonl  # noqa: E402

TARGET_TOLERANCE = 0.10


def words(text: str) -> int:
    return len(text.split())


def profile_arm(name: str) -> str:
    return f"length_matched_{name}"


def score_profiles(rows: list[dict]) -> tuple[dict, dict]:
    """Return (eq stats, per-profile stats)."""
    eq_values = [words(row["eq"]) for row in rows if isinstance(row.get("eq"), str) and row["eq"].strip()]
    if not eq_values:
        raise ValueError("No non-empty EQ arm to calibrate against.")
    eq_stats = {
        "n": len(eq_values),
        "mean_words": round(statistics.fmean(eq_values), 2),
        "median_words": round(statistics.median(eq_values), 2),
    }

    profiles: dict[str, dict] = {}
    for row in rows:
        for key, value in row.items():
            if not key.startswith("length_matched_") or not isinstance(value, str):
                continue
            if not value.strip():
                continue
            profiles.setdefault(key[len("length_matched_") :], []).append(words(value))

    stats: dict[str, dict] = {}
    for name, values in sorted(profiles.items()):
        mean = round(statistics.fmean(values), 2)
        median = round(statistics.median(values), 2)
        mean_gap = round(mean / eq_stats["mean_words"] - 1.0, 4)
        median_gap = round(median / eq_stats["median_words"] - 1.0, 4)
        stats[name] = {
            "n": len(values),
            "mean_words": mean,
            "median_words": median,
            "mean_gap_vs_eq": mean_gap,
            "median_gap_vs_eq": median_gap,
            "worst_gap": round(max(abs(mean_gap), abs(median_gap)), 4),
        }
    return eq_stats, stats


def choose(eq_stats: dict, stats: dict) -> dict:
    if not stats:
        raise ValueError("No length-matched profile arms found in the pairs file.")
    ranked = sorted(stats.items(), key=lambda kv: (kv[1]["worst_gap"], kv[0]))
    best_name, best = ranked[0]
    return {
        "selected_profile": best_name,
        "worst_gap": best["worst_gap"],
        "within_tolerance": best["worst_gap"] <= TARGET_TOLERANCE,
        "ranking": [{"profile": name, "worst_gap": row["worst_gap"]} for name, row in ranked],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs")
    parser.add_argument("--output")
    parser.add_argument("--tolerance", type=float, default=TARGET_TOLERANCE)
    args = parser.parse_args()

    rows = load_jsonl(args.pairs)
    eq_stats, stats = score_profiles(rows)
    selection = choose(eq_stats, stats)
    selection["within_tolerance"] = selection["worst_gap"] <= args.tolerance
    selection["tolerance"] = args.tolerance

    report = {
        "audit": "length-control-calibration",
        "n_cases": len(rows),
        "eq": eq_stats,
        "profiles": stats,
        "selection": selection,
        "method": (
            "Profiles are ranked by the larger of the mean and median relative gap "
            "to the EQ arm. Selection uses length only."
        ),
        "claim_boundary": (
            "A calibrated control arm removes length as a confound. It is a "
            "precondition for reading a preference test, not evidence that any arm "
            "is better. No ratings are involved."
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