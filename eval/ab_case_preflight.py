"""Preflight held-out response-level A/B cases before generation.

Usage:
  python eval/ab_case_preflight.py heldout.jsonl
  python eval/ab_case_preflight.py heldout.jsonl --final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LEAKAGE_FIELDS = {
    "expected_policy",
    "expected_intent",
    "expected_concepts",
    "annotated",
}


def load(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def validate(cases: list[dict], *, final: bool = False) -> dict:
    errors: list[str] = []
    seen: set[str] = set()
    role_counts = {"user": 0, "assistant": 0, "system": 0, "other": 0}

    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id", "")).strip()
        if not case_id:
            errors.append(f"line {index}: missing id")
            continue
        if case_id in seen:
            errors.append(f"{case_id}: duplicate id")
        seen.add(case_id)

        messages = case.get("transcript") or case.get("context")
        if not isinstance(messages, list) or not messages:
            errors.append(f"{case_id}: missing transcript/context")
            continue
        if not any(m.get("role") == "user" for m in messages if isinstance(m, dict)):
            errors.append(f"{case_id}: transcript contains no user turn")

        for message in messages:
            if not isinstance(message, dict):
                errors.append(f"{case_id}: non-object message")
                continue
            role = str(message.get("role", ""))
            role_counts[role if role in role_counts else "other"] += 1
            if not str(message.get("content", "")).strip():
                errors.append(f"{case_id}: empty message content")

        if final:
            leaked = sorted(LEAKAGE_FIELDS.intersection(case))
            if leaked:
                errors.append(
                    f"{case_id}: final held-out case contains development/gold fields: "
                    + ", ".join(leaked)
                )

    if final and len(cases) < 120:
        errors.append(
            f"final run has {len(cases)} cases; protocol target is at least 120 "
            "total cases to make ~90+ non-ties plausible"
        )

    return {
        "n_cases": len(cases),
        "unique_ids": len(seen),
        "role_counts": role_counts,
        "final_mode": final,
        "errors": errors,
        "ok": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()

    report = validate(load(args.cases), final=args.final)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
