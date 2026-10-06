"""Adversarial regression benchmark for short dialogue-reference state.

This benchmark is maintainer-authored and development-only. It is not an
external validation set, a semantic coreference benchmark, or evidence of
real-world response-quality improvement.

Usage: python eval/dialogue_reference_adversarial_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.dialogue_reference import infer_dialogue_reference  # noqa: E402


DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "dialogue_reference_adversarial.jsonl"
)


def _load_cases() -> list[dict]:
    rows: list[dict] = []
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc
    return rows


def _check(case: dict) -> tuple[list[str], dict]:
    state = infer_dialogue_reference(case["transcript"])
    expected = case["expected"]
    failures: list[str] = []

    for field in ("active", "resolved"):
        actual = getattr(state, field)
        if actual != expected[field]:
            failures.append(
                f"{field}: expected {expected[field]!r}, got {actual!r}"
            )

    if "form" in expected and state.form != expected["form"]:
        failures.append(
            f"form: expected {expected['form']!r}, got {state.form!r}"
        )

    if "target_role" in expected and state.target_role != expected["target_role"]:
        failures.append(
            "target_role: expected "
            f"{expected['target_role']!r}, got {state.target_role!r}"
        )

    if (
        "target_turn_index" in expected
        and state.target_turn_index != expected["target_turn_index"]
    ):
        failures.append(
            "target_turn_index: expected "
            f"{expected['target_turn_index']!r}, got "
            f"{state.target_turn_index!r}"
        )

    observed = {
        "active": state.active,
        "resolved": state.resolved,
        "form": state.form,
        "target_role": state.target_role,
        "target_turn_index": state.target_turn_index,
        "requires_clarification": state.requires_clarification,
        "evidence": list(state.evidence),
    }
    return failures, observed


def main() -> int:
    cases = _load_cases()
    if len(cases) < 24:
        raise AssertionError(
            "Adversarial reference benchmark must contain at least 24 cases."
        )

    failures: list[dict] = []
    class_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    passed_by_class: Counter[str] = Counter()
    passed_by_language: Counter[str] = Counter()
    unsafe_resolutions = 0
    missed_safe_resolutions = 0

    for case in cases:
        case_class = str(case["class"])
        language = str(case["language"])
        class_counts[case_class] += 1
        language_counts[language] += 1

        case_failures, observed = _check(case)
        expected_resolved = bool(case["expected"]["resolved"])

        if not expected_resolved and observed["resolved"]:
            unsafe_resolutions += 1
        if expected_resolved and not observed["resolved"]:
            missed_safe_resolutions += 1

        if case_failures:
            failures.append(
                {
                    "id": case["id"],
                    "class": case_class,
                    "language": language,
                    "failures": case_failures,
                    "observed": observed,
                }
            )
        else:
            passed_by_class[case_class] += 1
            passed_by_language[language] += 1

    passed = len(cases) - len(failures)
    report = {
        "benchmark": "dialogue-reference-adversarial-regression-v1",
        "status": (
            "pass" if not failures else "fail"
        ),
        "claim_boundary": (
            "Maintainer-authored development regression benchmark only; "
            "not external validation and not a semantic coreference benchmark."
        ),
        "cases": len(cases),
        "passed": passed,
        "exact_match_rate": passed / len(cases),
        "unsafe_resolutions": unsafe_resolutions,
        "missed_safe_resolutions": missed_safe_resolutions,
        "class_counts": dict(sorted(class_counts.items())),
        "passed_by_class": dict(sorted(passed_by_class.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "passed_by_language": dict(sorted(passed_by_language.items())),
        "failures": failures,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
