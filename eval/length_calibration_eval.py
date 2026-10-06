"""Check that profile selection uses length and nothing else.

The calibration decides which control arm stands in for the raw baseline. If it
were free to rank profiles on preference, the control would be fitted to the
outcome it is meant to be compared against, so these tests pin the selection
rule and its failure modes.

Usage: python eval/length_calibration_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.length_calibration import choose, score_profiles  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def row(eq: int, **arms: int) -> dict:
    return {
        "id": f"c{sum(arms.values())}{eq}",
        "eq": " ".join(["w"] * eq),
        **{f"length_matched_{name}": " ".join(["w"] * n) for name, n in arms.items()},
    }


def main() -> int:
    # EQ sits at 20 words. terse matches, generic overshoots by half.
    rows = [row(20, terse=20, generic=30) for _ in range(4)]
    eq_stats, stats = score_profiles(rows)
    check(eq_stats["mean_words"] == 20.0, f"EQ mean should be 20, got {eq_stats['mean_words']}")
    check(stats["terse"]["mean_gap_vs_eq"] == 0.0, "terse should match EQ exactly")
    check(
        stats["generic"]["mean_gap_vs_eq"] == 0.5,
        f"generic should overshoot by 0.5, got {stats['generic']['mean_gap_vs_eq']}",
    )

    picked = choose(eq_stats, stats)
    check(picked["selected_profile"] == "terse", f"terse should win, got {picked['selected_profile']}")
    check(picked["within_tolerance"] is True, "an exact match must be within tolerance")
    check(
        [r["profile"] for r in picked["ranking"]] == ["terse", "generic"],
        "ranking must be ordered by gap",
    )

    # A profile that matches the mean but not the median must not be selected.
    skewed = [
        {"id": "a", "eq": " ".join(["w"] * 10), "length_matched_p": " ".join(["w"] * 20)},
        {"id": "b", "eq": " ".join(["w"] * 10), "length_matched_p": " ".join(["w"] * 10)},
        {"id": "c", "eq": " ".join(["w"] * 90), "length_matched_p": " ".join(["w"] * 10)},
    ]
    eq2, stats2 = score_profiles(skewed)
    check(
        stats2["p"]["worst_gap"] >= stats2["p"]["mean_gap_vs_eq"],
        "worst_gap must account for the median too",
    )

    # Ties break deterministically by name so the choice is reproducible.
    tied_rows = [row(20, zeta=20, alpha=20) for _ in range(3)]
    eq3, stats3 = score_profiles(tied_rows)
    tied = choose(eq3, stats3)
    check(tied["selected_profile"] == "alpha", f"ties must break by name, got {tied['selected_profile']}")

    # An out-of-tolerance winner must be reported as such rather than silently accepted.
    far_rows = [row(20, terse=40) for _ in range(3)]
    eq4, stats4 = score_profiles(far_rows)
    far = choose(eq4, stats4)
    check(far["within_tolerance"] is False, "a 100% gap must not count as matched")
    check(far["worst_gap"] == 1.0, f"gap should be 1.0, got {far['worst_gap']}")

    # Empty or whitespace arms must not be counted as length data.
    dirty = [
        {"id": "a", "eq": "w w", "length_matched_p": "   "},
        {"id": "b", "eq": "w w w w", "length_matched_p": "w w w w"},
    ]
    eq5, stats5 = score_profiles(dirty)
    check(stats5["p"]["n"] == 1, f"blank arm must be dropped, got n={stats5['p']['n']}")

    # No EQ arm at all is an error, not a silent pass.
    try:
        score_profiles([{"id": "a", "eq": "", "length_matched_p": "w"}])
        FAILURES.append("empty EQ arm should raise")
    except ValueError:
        pass

    # No profile arm is an error too.
    try:
        choose(eq_stats, {})
        FAILURES.append("no profiles should raise")
    except ValueError:
        pass

    print(json.dumps({"failures": FAILURES, "ok": not FAILURES}, indent=2))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())