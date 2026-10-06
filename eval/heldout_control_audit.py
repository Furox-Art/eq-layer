"""Run the production EQ-Layer control path on held-out cases without LLM generation.

This produces generation-compatible metadata records so routing sensitivity can
be audited across the full external held-out set without spending model
inference or exposing human ratings.

Usage:
  python eval/heldout_control_audit.py heldout.jsonl control_records.jsonl \
    --affect-model artifacts/emobank_affect.joblib \
    --subtext-model artifacts/xdailydialog_subtext.joblib
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.ab_eval import load_jsonl, write_jsonl  # noqa: E402
from eq_layer.response_experiment import (  # noqa: E402
    FullEQPipeline,
    validate_experiment_cases,
)


def build_control_records(
    cases: list[dict],
    pipeline,
) -> list[dict]:
    records: list[dict] = []
    for case in cases:
        messages = list(case.get("transcript") or case.get("context") or [])
        _steered, metadata = pipeline.steer_messages(
            messages,
            annotated=None,
        )
        records.append(
            {
                "id": str(case["id"]),
                "source": case.get("source"),
                "generation": metadata,
            }
        )
    return records


def summarize(records: list[dict]) -> dict:
    actions = Counter()
    strata = Counter()
    clarify_by_stratum = Counter()
    repair_active = Counter()

    for record in records:
        generation = record.get("generation") or {}
        action = (generation.get("intent_risk_decision") or {}).get("action")
        if action:
            actions[str(action)] += 1

        source = record.get("source") or {}
        stratum = str(source.get("stratum") or "unstratified")
        strata[stratum] += 1
        if action == "clarify":
            clarify_by_stratum[stratum] += 1

        repair = generation.get("repair") or {}
        if repair.get("active"):
            repair_active[stratum] += 1

    return {
        "n_cases": len(records),
        "action_counts": dict(actions),
        "stratum_counts": dict(strata),
        "clarification_rate_by_stratum": {
            stratum: round(
                clarify_by_stratum.get(stratum, 0) / count,
                4,
            )
            for stratum, count in sorted(strata.items())
        },
        "active_repair_count_by_stratum": {
            stratum: repair_active.get(stratum, 0)
            for stratum in sorted(strata)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases")
    parser.add_argument("output")
    parser.add_argument("--affect-model", required=True)
    parser.add_argument("--subtext-model", required=True)
    parser.add_argument(
        "--final",
        action="store_true",
        help="Require final held-out preflight before auditing.",
    )
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    preflight = validate_experiment_cases(cases, final=args.final)
    if not preflight["ok"]:
        parser.error(
            "Held-out control audit preflight failed: "
            + " | ".join(preflight["errors"])
        )

    pipeline = FullEQPipeline.load(
        affect_model=args.affect_model,
        subtext_model=args.subtext_model,
    )
    records = build_control_records(cases, pipeline)
    write_jsonl(args.output, records)

    report = {
        "audit": "heldout-production-control-path",
        "preflight": preflight,
        **summarize(records),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
