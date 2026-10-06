"""Check that the length-matched arm isolates surface budget from policy.

Without this arm the EQ-vs-baseline comparison is confounded: on the frozen
120-case set the EQ arm came out 252 vs 79 characters against its own
baseline, so a preference could be length rather than steering. These tests
pin the invariants that make the third arm mean what it claims to mean.

Usage: python eval/length_matched_arm_eval.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.response_experiment import (  # noqa: E402
    LENGTH_CONTROL_INSTRUCTION,
    build_length_matched_messages,
    experiment_manifest,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def write_stub_dir() -> str:
    """experiment_manifest hashes the case and model files, so they must exist."""
    directory = tempfile.mkdtemp(prefix="eq-length-matched-")
    for name in ("cases.jsonl", "a.joblib", "s.joblib"):
        with open(os.path.join(directory, name), "wb") as handle:
            handle.write(name.encode("utf-8"))
    return directory


def main() -> int:
    transcript = [
        {"role": "user", "content": "Randevumu üç kez değiştirdiler"},
        {"role": "assistant", "content": "Anlıyorum"},
        {"role": "user", "content": "Ne yapacağım?"},
    ]

    # Budget is pinned from the deterministic realization, not invented.
    built = build_length_matched_messages(transcript, {"verbosity": "low", "question_budget": 1})
    check(built[0]["role"] == "system", "length-matched arm must prepend a system message")
    check(built[0]["content"] == LENGTH_CONTROL_INSTRUCTION.format(question_budget=1), "budget must come from realization")
    check("1 question" in built[0]["content"], "question budget must appear in the instruction")
    check(built[1:] == transcript, "original transcript must be preserved verbatim")
    check(
        len(built) == len(transcript) + 1,
        "exactly one message must be added, not a rewritten transcript",
    )

    # No EQ policy content may leak into the control arm.
    control_text = built[0]["content"].lower()
    for leak in ("empath", "affect", "policy", "mirror", "validate", "stance", "subtext"):
        check(leak not in control_text, f"control arm must not mention policy concept: {leak}")

    # A non-low verbosity still gets a budget instruction rather than nothing.
    normal = build_length_matched_messages(transcript, {"verbosity": "normal", "question_budget": 0})
    check(normal[0]["role"] == "system", "normal verbosity must still produce an instruction")
    check("0 question" in normal[0]["content"], "zero budget must still be stated")

    # Manifest must not let the three-arm design be read as a paired design.
    stub_dir = write_stub_dir()
    cases_path = os.path.join(stub_dir, "cases.jsonl")
    affect_model = os.path.join(stub_dir, "a.joblib")
    subtext_model = os.path.join(stub_dir, "s.joblib")

    manifest = experiment_manifest(
        n_cases=12,
        cases_path=cases_path,
        affect_model=affect_model,
        subtext_model=subtext_model,
        model=FakeModel(),
        global_seed=42,
        allow_annotations=False,
        length_matched_arm=True,
    )
    check(manifest["length_matched_arm"] is True, "manifest must record the arm")
    check(manifest["arms"] == ["baseline", "eq", "length_matched"], f"arms should list three, got {manifest['arms']}")
    check("three-arm" in manifest["design"], "design should name the three-arm layout")

    paired = experiment_manifest(
        n_cases=12,
        cases_path=cases_path,
        affect_model=affect_model,
        subtext_model=subtext_model,
        model=FakeModel(),
        global_seed=42,
        allow_annotations=False,
    )
    check(paired["arms"] == ["baseline", "eq"], "without the flag the design stays two-arm")
    check("claim_boundary" in manifest, "three-arm manifest must carry a claim boundary")
    check(
        "isolates" in manifest["length_control"],
        "length_control must state that EQ-vs-length_matched is the clean comparison",
    )

    shutil.rmtree(stub_dir, ignore_errors=True)
    print(json.dumps({"failures": FAILURES, "ok": not FAILURES}, indent=2))
    return 1 if FAILURES else 0


class FakeModel:
    model_id = "test-model"
    temperature = 0.0
    command = ["python", "server.py"]


if __name__ == "__main__":
    raise SystemExit(main())