"""Deterministic held-out routing evaluation.

Measures the layer's own decision instead of a rendered response. The
generative A/B path cannot answer whether EQ-Layer helps: the EQ arm comes out
~3x shorter than the baseline arm (252 vs 79 characters, 3.19x), so any
preference is confounded with length before a single rating is collected.

`FactoredAction` removes the confound because the surface form is an explicit
control rather than a side effect of how much the model wrote:

    verbosity, directness, warmth, question_budget, scope_limited, no_guess

So the decision under test is `task_move` / `social_move` / `repair_move`, and
length is a pinned parameter rather than an accidental treatment effect.

This script has no model backend and no network access. It emits the routing
decision per held-out case plus a one-label-per-case adjudication ballot, which
is ~120 human labels instead of ~1800 pairwise ratings.

Usage:
    python eval/route_heldout_eval.py \
        --cases artifacts/ab_heldout_120.jsonl \
        --output eval/results/deterministic_routing_120.json \
        --ballot artifacts/route_gold_ballot.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.response_experiment import FullEQPipeline  # noqa: E402

# The distinct decisions the adjudicator chooses between. Kept small on
# purpose: one label per case has to be cheap enough that it actually gets
# done, and every extra option multiplies rater disagreement.
ADJUDICATION_MOVES = (
    "execute_request",
    "clarify_goal",
    "answer_question",
    "explain",
    "respond_contextually",
    "report_status",
    "set_boundary",
)

GOLD_QUESTIONS = (
    "intent",  # what does the user actually want done
    "task_move",  # the task move
    "surface",  # how much should the reply say
)


def load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def route_case(pipeline: FullEQPipeline, case: dict) -> dict:
    messages = case.get("transcript") or case.get("context") or []
    _, metadata = pipeline.steer_messages(messages)
    action = metadata.get("factored_action") or {}
    realization = action.get("realization") or {}
    return {
        "id": case["id"],
        "stratum": (case.get("source") or {}).get("stratum"),
        "policy": metadata.get("policy"),
        "register": metadata.get("register"),
        "intent_kind": metadata.get("intent_kind"),
        "intent_confidence": metadata.get("intent_confidence"),
        "factored_action": action,
        "task_move": action.get("task_move"),
        "social_move": action.get("social_move"),
        "repair_move": action.get("repair_move"),
        "verbosity": realization.get("verbosity"),
        "question_budget": realization.get("question_budget"),
    }


def summarise(rows: list[dict]) -> dict:
    def counts(key: str) -> dict:
        return dict(sorted(Counter(row[key] for row in rows if row.get(key)).items()))

    strata: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("stratum"):
            strata.setdefault(row["stratum"], []).append(row)

    return {
        "n_cases": len(rows),
        "task_move": counts("task_move"),
        "social_move": counts("social_move"),
        "repair_move": counts("repair_move"),
        "intent_kind": counts("intent_kind"),
        "policy": counts("policy"),
        "verbosity": counts("verbosity"),
        "question_budget": counts("question_budget"),
        "by_stratum": {
            stratum: {
                "n": len(group),
                "task_move": dict(sorted(Counter(r["task_move"] for r in group).items())),
            }
            for stratum, group in sorted(strata.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ballot")
    parser.add_argument("--affect-model")
    parser.add_argument("--subtext-model")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if bool(args.affect_model) != bool(args.subtext_model):
        parser.error("--affect-model and --subtext-model must be supplied together")
    if not args.affect_model:
        parser.error(
            "trained components are required: pass --affect-model and --subtext-model. "
            "The deterministic routing claim is about the learned components, not the "
            "keyword fallback."
        )

    pipeline = FullEQPipeline.load(args.affect_model, args.subtext_model)
    rows = [route_case(pipeline, case) for case in cases]

    summary = summarise(rows)
    summary["measurement"] = {
        "unit": "factored_action_decision",
        "llm_in_path": False,
        "length_control": "verbosity/question_budget are explicit realization controls",
        "adjudication_moves": list(ADJUDICATION_MOVES),
        "gold_questions": list(GOLD_QUESTIONS),
    }
    summary["claim_boundary"] = (
        "Routing distribution is a descriptive audit of the deterministic decision. "
        "It is not evidence that responses are better; agreement with adjudicated "
        "task moves has to be measured separately, and response quality is out of "
        "scope for the deterministic path."
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.ballot:
        ballot_path = Path(args.ballot)
        ballot_path.parent.mkdir(parents=True, exist_ok=True)
        with ballot_path.open("w", encoding="utf-8") as handle:
            for case, row in zip(cases, rows, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "id": case["id"],
                            "context": case.get("transcript") or case.get("context"),
                            "eq_layer_decision": {
                                "task_move": row["task_move"],
                                "social_move": row["social_move"],
                                "repair_move": row["repair_move"],
                                "realization": row["realization"],
                            },
                            "adjudication": {
                                "acceptable_task_moves": list(ADJUDICATION_MOVES),
                                "acceptable_task_move": "",
                                "surface_amount": "",
                                "notes": "",
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())