"""Check the arm length report reads the right arms and survives bad rows.

The report is the thing that will be looked at to decide whether the control
arm worked, so it has to surface a missing or empty arm rather than quietly
averaging over fewer cases than the run claims.

Usage: python eval/length_report_eval.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def run_report(rows: list[dict]) -> dict:
    directory = tempfile.mkdtemp(prefix="eq-length-report-")
    path = os.path.join(directory, "pairs.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    completed = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "length_report.py"), path],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(f"length_report.py failed: {completed.stderr}")
    return json.loads(completed.stdout)


def main() -> int:
    # EQ is 3x shorter than baseline; the control should land near EQ.
    rows = [
        {"id": "a", "baseline": "w " * 90, "eq": "w " * 30, "length_matched": "w " * 30},
        {"id": "b", "baseline": "w " * 60, "eq": "w " * 20, "length_matched": "w " * 20},
    ]
    report = run_report(rows)

    check(report["n_cases"] == 2, "n_cases should count rows")
    check(report["arms_present"] == ["baseline", "eq", "length_matched"], "all three arms should be found")
    # 30/90 and 20/60 words: both rows are exactly one third.
    check(
        abs(report["ratios"]["eq_vs_baseline"]["words"] - 0.333) < 0.005,
        f"eq_vs_baseline should be ~0.333, got {report['ratios']['eq_vs_baseline']['words']}",
    )
    check(
        report["ratios"]["eq_vs_length_matched"]["words"] == 1.0,
        f"eq_vs_length_matched should be 1.0, got {report['ratios']['eq_vs_length_matched']['words']}",
    )
    check(
        report["cases_missing_an_arm"] == [] and report["cases_with_empty_arm"] == [],
        "clean fixture should report no problems",
    )

    # A two-arm file must not pretend the control exists.
    two_arm = run_report([{"id": "a", "baseline": "w " * 10, "eq": "w " * 5}])
    check(two_arm["arms_present"] == ["baseline", "eq"], "two-arm file should list two arms")
    check(
        "eq_vs_length_matched" not in two_arm["ratios"],
        "ratio to a non-existent arm must be absent, not guessed",
    )
    check(
        "eq_vs_baseline" in two_arm["ratios"],
        "the original comparison must still work without the control",
    )

    # Missing and empty arms must be named.
    dirty = run_report(
        [
            {"id": "a", "baseline": "w " * 10, "eq": "w " * 5, "length_matched": "w " * 5},
            {"id": "b", "baseline": "w " * 10},
            {"id": "c", "baseline": "", "eq": "w " * 5, "length_matched": "w " * 5},
        ]
    )
    check(dirty["cases_missing_an_arm"] == ["b"], f"missing arm should be named, got {dirty['cases_missing_an_arm']}")
    check(dirty["cases_with_empty_arm"] == ["c"], f"empty arm should be named, got {dirty['cases_with_empty_arm']}")
    # baseline is present in all three rows: a has it, b has it but no eq, c has
    # it as an empty string. An empty string is still a present arm, which is
    # exactly why cases_with_empty_arm has to name it.
    check(
        dirty["totals"]["words"]["baseline"]["n"] == 3,
        f"baseline present in all 3 rows, got n={dirty['totals']['words']['baseline']['n']}",
    )
    check(
        dirty["totals"]["words"]["eq"]["n"] == 2,
        f"eq present in a and c only, got n={dirty['totals']['words']['eq']['n']}",
    )

    print(json.dumps({"failures": FAILURES, "ok": not FAILURES}, indent=2))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())