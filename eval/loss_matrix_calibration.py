"""Estimate a candidate intent loss matrix from independent human cost ratings.

This utility never promotes a matrix to production. Its output is a transparent
candidate that still requires an independent validation split.

Input JSONL row:
{
  "case_id": "cal-001",
  "source_id": "optional-upstream-id",
  "rater_id": "r1",
  "true_intent": "action_request",
  "decision_costs": {
    "action_request": 0,
    "status_check": 3,
    "explanation": 3,
    "question": 2,
    "statement": 4,
    "clarify": 1
  }
}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from eq_layer.intent_belief import DECISIONS, INTENT_LABELS


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            rows.append(row)
    return rows


def forbidden_identity_sets(paths: list[str]) -> tuple[set[str], set[str]]:
    case_ids: set[str] = set()
    source_ids: set[str] = set()
    for path in paths:
        for row in load_jsonl(path):
            case_id = str(row.get("id") or row.get("case_id") or "").strip()
            if case_id:
                case_ids.add(case_id)
            source = row.get("source")
            if isinstance(source, dict):
                source_id = str(source.get("source_id") or "").strip()
                if source_id:
                    source_ids.add(source_id)
            direct_source = str(row.get("source_id") or "").strip()
            if direct_source:
                source_ids.add(direct_source)
    return case_ids, source_ids


def estimate_candidate(
    rows: list[dict],
    *,
    forbidden_case_ids: set[str] | None = None,
    forbidden_source_ids: set[str] | None = None,
    min_raters_per_case: int = 3,
    min_cases_per_intent: int = 1,
) -> dict:
    if min_raters_per_case < 1:
        raise ValueError("min_raters_per_case must be >= 1")
    if min_cases_per_intent < 1:
        raise ValueError("min_cases_per_intent must be >= 1")

    forbidden_case_ids = forbidden_case_ids or set()
    forbidden_source_ids = forbidden_source_ids or set()

    seen_pairs: set[tuple[str, str]] = set()
    intent_by_case: dict[str, str] = {}
    raters_by_case: dict[str, set[str]] = defaultdict(set)
    source_by_case: dict[str, str] = {}
    values: dict[tuple[str, str], list[float]] = defaultdict(list)

    expected_decisions = set(DECISIONS)

    for index, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id") or "").strip()
        rater_id = str(row.get("rater_id") or "").strip()
        true_intent = str(row.get("true_intent") or "").strip()
        source_id = str(row.get("source_id") or "").strip()

        if not case_id or not rater_id:
            raise ValueError(f"row {index}: case_id and rater_id are required")
        if case_id in forbidden_case_ids:
            raise ValueError(f"row {index}: forbidden A/B case overlap: {case_id}")
        if source_id and source_id in forbidden_source_ids:
            raise ValueError(
                f"row {index}: forbidden upstream source overlap: {source_id}"
            )
        if true_intent not in INTENT_LABELS:
            raise ValueError(
                f"row {index}: unsupported true_intent {true_intent!r}"
            )

        pair = (case_id, rater_id)
        if pair in seen_pairs:
            raise ValueError(
                f"row {index}: duplicate case/rater pair {case_id}/{rater_id}"
            )
        seen_pairs.add(pair)

        prior_intent = intent_by_case.get(case_id)
        if prior_intent is not None and prior_intent != true_intent:
            raise ValueError(
                f"row {index}: inconsistent adjudicated intent for {case_id}: "
                f"{prior_intent!r} vs {true_intent!r}"
            )
        intent_by_case[case_id] = true_intent
        raters_by_case[case_id].add(rater_id)
        if source_id:
            prior_source = source_by_case.get(case_id)
            if prior_source is not None and prior_source != source_id:
                raise ValueError(
                    f"row {index}: inconsistent source_id for {case_id}"
                )
            source_by_case[case_id] = source_id

        costs = row.get("decision_costs")
        if not isinstance(costs, dict):
            raise ValueError(f"row {index}: decision_costs must be an object")
        provided = set(costs)
        if provided != expected_decisions:
            missing = sorted(expected_decisions - provided)
            extra = sorted(provided - expected_decisions)
            raise ValueError(
                f"row {index}: decision_costs keys mismatch; "
                f"missing={missing}, extra={extra}"
            )

        for decision in DECISIONS:
            raw_cost = costs[decision]
            if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
                raise ValueError(
                    f"row {index}: {decision} cost must be numeric"
                )
            cost = float(raw_cost)
            if cost < 0.0 or cost > 4.0:
                raise ValueError(
                    f"row {index}: {decision} cost {cost} outside 0..4"
                )
            values[(decision, true_intent)].append(cost)

    if not rows:
        raise ValueError("No calibration ratings supplied")

    for case_id, raters in sorted(raters_by_case.items()):
        if len(raters) < min_raters_per_case:
            raise ValueError(
                f"{case_id}: only {len(raters)} raters; "
                f"need >= {min_raters_per_case}"
            )

    cases_by_intent: dict[str, set[str]] = {
        intent: set() for intent in INTENT_LABELS
    }
    for case_id, true_intent in intent_by_case.items():
        cases_by_intent[true_intent].add(case_id)

    underfilled = {
        intent: len(case_ids)
        for intent, case_ids in cases_by_intent.items()
        if len(case_ids) < min_cases_per_intent
    }
    if underfilled:
        raise ValueError(
            "insufficient cases per intent: "
            + ", ".join(
                f"{intent}={count}" for intent, count in sorted(underfilled.items())
            )
        )

    matrix: dict[str, dict[str, float]] = {}
    cell_counts: dict[str, dict[str, int]] = {}
    for decision in DECISIONS:
        matrix[decision] = {}
        cell_counts[decision] = {}
        for true_intent in INTENT_LABELS:
            cell = values[(decision, true_intent)]
            if not cell:
                raise ValueError(
                    f"empty decision/intent cell: {decision}/{true_intent}"
                )
            matrix[decision][true_intent] = fmean(cell) / 4.0
            cell_counts[decision][true_intent] = len(cell)

    return {
        "status": "candidate-only-not-production",
        "matrix_id": "human-calibration-candidate-v1",
        "scale": {
            "input": "ordinal interaction cost 0..4",
            "output": "mean cost divided by 4",
        },
        "n_rating_rows": len(rows),
        "n_cases": len(intent_by_case),
        "raters_per_case": {
            case_id: len(raters)
            for case_id, raters in sorted(raters_by_case.items())
        },
        "cases_per_true_intent": {
            intent: len(cases_by_intent[intent])
            for intent in INTENT_LABELS
        },
        "cell_rating_counts": cell_counts,
        "loss_matrix": matrix,
        "guardrail": (
            "Candidate only. Do not replace hand-specified-v1 until a frozen, "
            "independent validation split passes the preregistered calibration "
            "protocol. Blinded A/B response-preference outcomes are forbidden "
            "tuning data."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations")
    parser.add_argument("output")
    parser.add_argument(
        "--forbid-cases",
        action="append",
        default=[],
        help=(
            "JSONL file whose case/source identities are forbidden calibration "
            "overlap. Repeat for multiple files."
        ),
    )
    parser.add_argument("--min-raters-per-case", type=int, default=3)
    parser.add_argument("--min-cases-per-intent", type=int, default=1)
    args = parser.parse_args()

    rows = load_jsonl(args.annotations)
    forbidden_cases, forbidden_sources = forbidden_identity_sets(args.forbid_cases)
    report = estimate_candidate(
        rows,
        forbidden_case_ids=forbidden_cases,
        forbidden_source_ids=forbidden_sources,
        min_raters_per_case=args.min_raters_per_case,
        min_cases_per_intent=args.min_cases_per_intent,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
