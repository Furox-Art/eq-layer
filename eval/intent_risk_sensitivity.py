"""Audit Bayes-risk routing sensitivity on generated A/B pairs.

This is a development/audit utility. It does not change production routing and
must not be used to tune on blinded human outcomes.

Usage:
  python eval/intent_risk_sensitivity.py pairs.jsonl \
    --heldout heldout.jsonl \
    --output sensitivity.json

The audit:
- reads the raw and fused intent posteriors stored in generation metadata;
- recomputes one-step Bayes-risk actions under clarification-cost multipliers;
- reports action flips, clarification rates, entropy and source-stratum counts;
- preserves the production loss matrix unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.intent_belief import DECISIONS, INTENT_LABELS, LOSS_MATRIX


DEFAULT_MULTIPLIERS = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def scaled_loss_matrix(clarify_multiplier: float) -> dict[str, dict[str, float]]:
    if clarify_multiplier <= 0:
        raise ValueError("clarify multiplier must be > 0")
    matrix = {
        decision: dict(costs)
        for decision, costs in LOSS_MATRIX.items()
    }
    matrix["clarify"] = {
        label: cost * clarify_multiplier
        for label, cost in LOSS_MATRIX["clarify"].items()
    }
    return matrix


def decide_from_mapping(
    belief: dict[str, float],
    *,
    clarify_multiplier: float = 1.0,
) -> tuple[str, float]:
    matrix = scaled_loss_matrix(clarify_multiplier)
    probabilities = {
        label: max(0.0, float(belief.get(label, 0.0)))
        for label in INTENT_LABELS
    }
    total = sum(probabilities.values())
    if total <= 0:
        probabilities = {
            label: (1.0 if label == "unknown" else 0.0)
            for label in INTENT_LABELS
        }
    else:
        probabilities = {
            label: value / total
            for label, value in probabilities.items()
        }

    losses = {
        decision: sum(
            probabilities[label] * matrix[decision][label]
            for label in INTENT_LABELS
        )
        for decision in DECISIONS
    }
    best = min(losses.values())
    winners = [
        decision
        for decision, loss in losses.items()
        if abs(loss - best) <= 1e-12
    ]
    chosen = (
        "clarify"
        if "clarify" in winners
        else next(decision for decision in DECISIONS if decision in winners)
    )
    return chosen, round(losses[chosen], 6)


def source_map(heldout: list[dict] | None) -> dict[str, dict]:
    if not heldout:
        return {}
    return {
        str(row["id"]): dict(row.get("source") or {})
        for row in heldout
    }


def audit(
    pairs: list[dict],
    *,
    heldout: list[dict] | None = None,
    multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS,
) -> dict:
    sources = source_map(heldout)
    rows = []

    for pair in pairs:
        case_id = str(pair["id"])
        generation = pair.get("generation") or {}
        source = pair.get("source") or sources.get(case_id) or {}
        fused = generation.get("intent_belief")
        raw = generation.get("intent_belief_raw")

        if not isinstance(fused, dict):
            raise ValueError(f"{case_id}: missing generation.intent_belief")

        base_action, base_loss = decide_from_mapping(fused)
        recorded = (
            generation.get("intent_risk_decision") or {}
        ).get("action")
        if recorded is not None and recorded != base_action:
            raise ValueError(
                f"{case_id}: stored action={recorded} but recomputed={base_action}"
            )

        raw_action = None
        if isinstance(raw, dict):
            raw_action, _ = decide_from_mapping(raw)

        rows.append(
            {
                "id": case_id,
                "stratum": source.get("stratum") or "unstratified",
                "raw_action": raw_action,
                "fused_action": base_action,
                "fused_expected_loss": base_loss,
                "entropy": generation.get("intent_belief_entropy"),
                "actions_by_multiplier": {
                    str(multiplier): decide_from_mapping(
                        fused,
                        clarify_multiplier=multiplier,
                    )[0]
                    for multiplier in multipliers
                },
            }
        )

    baseline_actions = Counter(row["fused_action"] for row in rows)
    raw_actions = Counter(
        row["raw_action"]
        for row in rows
        if row["raw_action"] is not None
    )

    by_multiplier = {}
    for multiplier in multipliers:
        key = str(multiplier)
        actions = [row["actions_by_multiplier"][key] for row in rows]
        flips = sum(
            action != row["fused_action"]
            for action, row in zip(actions, rows, strict=True)
        )
        by_multiplier[key] = {
            "action_counts": dict(Counter(actions)),
            "clarification_rate": (
                round(actions.count("clarify") / len(actions), 4)
                if actions
                else 0.0
            ),
            "flips_vs_production": flips,
        }

    strata = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row)

    by_stratum = {}
    for stratum, items in sorted(strata.items()):
        actions = [item["fused_action"] for item in items]
        raw = [
            item["raw_action"]
            for item in items
            if item["raw_action"] is not None
        ]
        entropies = [
            float(item["entropy"])
            for item in items
            if item["entropy"] is not None
        ]
        by_stratum[stratum] = {
            "n": len(items),
            "fused_action_counts": dict(Counter(actions)),
            "raw_action_counts": dict(Counter(raw)),
            "clarification_rate": round(
                actions.count("clarify") / len(actions),
                4,
            ),
            "mean_fused_entropy": (
                round(sum(entropies) / len(entropies), 4)
                if entropies
                else None
            ),
        }

    raw_to_fused_flips = sum(
        row["raw_action"] is not None
        and row["raw_action"] != row["fused_action"]
        for row in rows
    )

    return {
        "audit": "intent-risk-sensitivity-development-only",
        "n_cases": len(rows),
        "production_action_counts": dict(baseline_actions),
        "production_clarification_rate": (
            round(baseline_actions.get("clarify", 0) / len(rows), 4)
            if rows
            else 0.0
        ),
        "raw_action_counts": dict(raw_actions),
        "raw_to_fused_action_flips": raw_to_fused_flips,
        "by_stratum": by_stratum,
        "clarify_cost_sensitivity": by_multiplier,
        "interpretation_guardrail": (
            "This audit measures routing sensitivity only. It does not identify "
            "an optimal loss matrix and must not be tuned against blinded human "
            "preference outcomes."
        ),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs")
    parser.add_argument("--heldout")
    parser.add_argument("--output")
    parser.add_argument(
        "--multipliers",
        default=",".join(str(value) for value in DEFAULT_MULTIPLIERS),
    )
    args = parser.parse_args()

    multipliers = tuple(
        float(value)
        for value in args.multipliers.split(",")
        if value.strip()
    )
    pairs = load_jsonl(args.pairs)
    heldout = load_jsonl(args.heldout) if args.heldout else None
    report = audit(
        pairs,
        heldout=heldout,
        multipliers=multipliers,
    )

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
