"""Build the blinded ballot for the primary endpoint.

The primary endpoint is EQ against the calibrated length-matched control, not
EQ against the raw baseline. Against the raw baseline the EQ arm runs several
times shorter, so a preference would partly measure length. The decode key
names the control arm explicitly instead of calling it "baseline".

Profile selection happens inside the same run at full scale. A profile
calibrated on a 12-case pilot is not evidence that it holds on 120 cases, so
the winner is re-derived here rather than hard-coded.

Usage:
  python eval/prepare_selected_ballot.py \
    pairs.jsonl calibration.json ballot.jsonl key.json --seed 4242
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
    parser.add_argument("calibration")
    parser.add_argument("ballot")
    parser.add_argument("key")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--require-match",
        action="store_true",
        help="Fail if no profile landed within the calibration tolerance.",
    )
    args = parser.parse_args()

    with open(args.calibration, encoding="utf-8") as handle:
        calibration = json.load(handle)
    selection = calibration["selection"]
    profile = selection["selected_profile"]
    arm = f"length_matched:{profile}"

    if args.require_match and not selection.get("within_tolerance"):
        raise SystemExit(
            f"Selected profile {profile!r} has worst gap {selection['worst_gap']}, "
            f"outside the {selection['tolerance']} tolerance. The primary endpoint "
            "is not length-controlled; do not spend human ratings on it until it is."
        )

    rows = load_jsonl(args.pairs)
    pairs = []
    missing = []
    for row in rows:
        control = row.get(f"length_matched_{profile}")
        if not isinstance(control, str) or not control.strip():
            missing.append(str(row.get("id")))
            continue
        pairs.append(
            {
                "id": row["id"],
                "source": row.get("source"),
                "context": row.get("context", []),
                "baseline": control,
                "eq": row["eq"],
            }
        )

    if missing:
        raise SystemExit(f"Missing {arm} response for cases: {missing}")

    ballot, key = prepare_blinded(
        pairs,
        seed=args.seed,
        arm_names=("eq", arm),
    )

    write_jsonl(args.ballot, ballot)
    os.makedirs(os.path.dirname(os.path.abspath(args.key)), exist_ok=True)
    with open(args.key, "w", encoding="utf-8") as handle:
        json.dump(key, handle, indent=2, sort_keys=True)
        handle.write("\n")

    eq_first = sum(1 for row in key.values() if row["A"] == "eq")
    report = {
        "primary_endpoint": "eq vs length-matched control",
        "selected_profile": profile,
        "control_arm": arm,
        "worst_gap": selection["worst_gap"],
        "within_tolerance": selection.get("within_tolerance"),
        "n_ballot_rows": len(ballot),
        "eq_first_slots": eq_first,
        "control_first_slots": len(ballot) - eq_first,
        "comparison_not_balloted": {
            "arm": "baseline",
            "reason": (
                "EQ runs several times shorter than the raw baseline, so that "
                "comparison is confounded. It remains available in the pairs file "
                "for the descriptive length report."
            ),
        },
        "claim_boundary": (
            "Length is controlled; quality is not yet measured. A blank ballot "
            "carries no evidence."
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())