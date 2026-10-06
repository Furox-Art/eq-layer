"""Check that circularity exclusions actually change the scored numbers.

The exclusion logic is easy to write and easy to write wrong: a flag that
looks respected but filters nothing still produces a plausible report. These
tests assert the arithmetic, not the intent.

Usage: python eval/route_score_eval.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.ab_score import circular_ids  # noqa: E402
from eval.route_score import agreement, collect, report  # noqa: E402

FAILURES: list[str] = []
MOVES = ["execute_request", "clarify_goal", "answer_question", "respond_contextually"]


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def ballot_row(case_id: str, stratum: str, predicted: str, adjudicated: str, intent: str = "statement") -> dict:
    return {
        "id": case_id,
        "stratum": stratum,
        "context": [],
        "eq_layer_decision": {
            "task_move": predicted,
            "social_move": "mirror_specific",
            "repair_move": "none",
            "intent_kind": intent,
        },
        "adjudication": {
            "circular_stratum": stratum == "task_repair",
            "excluded_from_headline": stratum == "task_repair",
            "acceptable_task_moves": MOVES,
            "acceptable_task_move": adjudicated,
            "surface_amount": "",
            "notes": "",
        },
    }


def main() -> int:
    rows = [
        # task_repair agrees 2/2 — circular, must not reach the headline.
        ballot_row("c1", "task_repair", "execute_request", "execute_request"),
        ballot_row("c2", "task_repair", "clarify_goal", "clarify_goal"),
        # empathetic 3/4 agree.
        ballot_row("e1", "empathetic", "respond_contextually", "respond_contextually"),
        ballot_row("e2", "empathetic", "respond_contextually", "respond_contextually"),
        ballot_row("e3", "empathetic", "respond_contextually", "respond_contextually"),
        ballot_row("e4", "empathetic", "clarify_goal", "answer_question"),
        # task_general 1/2 agree, one case routes on unknown intent.
        ballot_row("g1", "task_general", "execute_request", "execute_request"),
        ballot_row("g2", "task_general", "clarify_goal", "clarify_goal", intent="unknown"),
        ballot_row("g3", "task_general", "answer_question", "execute_request", intent="unknown"),
    ]

    rated, problems = collect(rows)
    check(len(rated) == 9, f"expected 9 rated rows, got {len(rated)}")
    check(problems == [], "no row should be reported unrated")

    default = report(rated, problems, include_circular=False, include_unknown_intent=False)
    # Non-circular AND known intent: e1..e4, g1, (g2/g3 dropped).
    check(default["headline"]["n"] == 5, f"headline n should be 5, got {default['headline']['n']}")
    check(default["headline"]["agreed"] == 4, f"headline agreed should be 4, got {default['headline']['agreed']}")
    check(
        default["headline"]["agreement_rate"] == 0.8,
        f"headline rate should be 0.8, got {default['headline']['agreement_rate']}",
    )
    check(default["all_rated"]["n"] == 9, "all_rated should cover every row")
    check(
        default["all_rated"]["agreed"] == 7,
        f"all_rated agreed should be 7, got {default['all_rated']['agreed']}",
    )
    check(
        default["all_rated"]["agreement_rate"] == 0.7778,
        f"all_rated rate should be 0.7778, got {default['all_rated']['agreement_rate']}",
    )
    check(
        default["exclusions_applied"]["circular_strata_excluded"] is True
        and default["exclusions_applied"]["unknown_intent_excluded"] is True,
        "both exclusions should be recorded as applied",
    )

    # Including circular cases must move the headline, otherwise the flag is decorative.
    with_circular = report(rated, problems, include_circular=True, include_unknown_intent=False)
    check(with_circular["headline"]["n"] == 7, "including circular should add 2 rows")
    check(
        with_circular["headline"]["agreed"] == 6,
        f"including circular should raise agreed to 6, got {with_circular['headline']['agreed']}",
    )
    check(
        with_circular["headline"]["agreement_rate"] == 0.8571,
        f"circular-inclusive rate should be 0.8571, got {with_circular['headline']['agreement_rate']}",
    )

    with_unknown = report(rated, problems, include_circular=False, include_unknown_intent=True)
    check(with_unknown["headline"]["n"] == 7, "including unknown intent should restore 2 rows")
    check(
        with_unknown["headline"]["agreed"] == 5,
        f"unknown-inclusive agreed should be 5, got {with_unknown['headline']['agreed']}",
    )
    check(
        with_unknown["headline"]["agreement_rate"] == 0.7143,
        f"unknown-inclusive rate should be 0.7143, got {with_unknown['headline']['agreement_rate']}",
    )

    # Per-stratum reporting must still show the circular stratum.
    check("task_repair" in default["by_stratum"], "circular stratum should still be reported")
    check(
        default["by_stratum"]["task_repair"]["agreement_rate"] == 1.0,
        "task_repair fixture agrees fully, which is the point: it must not be the headline",
    )

    # Unrated rows are separated, not silently counted as disagreements.
    unrated = [ballot_row("u1", "empathetic", "clarify_goal", "")]
    rated2, problems2 = collect(rows + unrated)
    rep = report(rated2, problems2, include_circular=False, include_unknown_intent=False)
    check(rep["n_unrated"] == 1, "unrated row should be counted separately")
    check(rep["unrated_ids"] == ["u1"], "unrated id should be reported")
    check(rep["headline"]["n"] == 5, "unrated rows must not enter the headline")

    # Wilson interval sanity.
    w = agreement(3, 4)
    check(w["wilson_95_low"] is not None and w["wilson_95_low"] <= w["agreement_rate"], "Wilson low must sit at or below the rate")
    check(w["wilson_95_high"] is not None and w["wilson_95_high"] >= w["agreement_rate"], "Wilson high must sit at or above the rate")
    check(0.0 < w["wilson_95_low"] and w["wilson_95_high"] < 1.0, "Wilson interval must not touch 0 or 1 at n=4")
    check(agreement(0, 0)["agreement_rate"] is None, "empty set must not divide by zero")
    check(agreement(0, 0)["wilson_95_low"] is None, "empty set must not produce an interval")

    # An adjudicated move outside the allowed set is an error, not a disagreement.
    try:
        collect([ballot_row("bad", "empathetic", "clarify_goal", "set_boundary")])
        FAILURES.append("out-of-set adjudicated move should raise")
    except ValueError:
        pass

    # Pairwise scorer must find circular ids from the stratum field.
    pairwise = [
        {"id": "c1", "stratum": "task_repair", "ratings": {}},
        {"id": "e1", "stratum": "empathetic", "ratings": {}},
        {"id": "e2", "source": {"stratum": "task_repair"}, "ratings": {}},
        {"id": "e3", "stratum": None, "ratings": {}},
    ]
    check(
        circular_ids(pairwise) == {"c1", "e2"},
        f"circular_ids should find nested and top-level strata, got {circular_ids(pairwise)}",
    )

    print(json.dumps({"failures": FAILURES, "ok": not FAILURES}, indent=2))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())