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
    DEFAULT_LENGTH_PROFILE,
    LENGTH_CONTROL_PROFILES,
    build_length_matched_messages,
    experiment_manifest,
)

FAILURES: list[str] = []

# A profile phrased as prohibitions made replies longer on a small model. See the
# regression guard in main().
NEGATIVE_MARKERS = ("no ", "not ", "never", "avoid", "without ", "don't", "do not", "cannot")


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
    built = build_length_matched_messages(
        transcript, {"verbosity": "low", "question_budget": 1}, "answer_only"
    )
    check(built[0]["role"] == "system", "length-matched arm must prepend a system message")
    check(
        built[0]["content"] == LENGTH_CONTROL_PROFILES["answer_only"].format(question_budget=1),
        "budget must come from the realization",
    )
    check("1 question" in built[0]["content"], "question budget must appear in the instruction")
    check(built[1:] == transcript, "original transcript must be preserved verbatim")
    check(
        len(built) == len(transcript) + 1,
        "exactly one message must be added, not a rewritten transcript",
    )

    # No EQ policy content may leak into any control profile.
    for name in LENGTH_CONTROL_PROFILES:
        arm = build_length_matched_messages(
            transcript, {"verbosity": "low", "question_budget": 1}, name
        )
        control_text = arm[0]["content"].lower()
        for leak in ("empath", "affect", "policy", "mirror", "validate", "stance", "subtext"):
            check(
                leak not in control_text,
                f"profile {name} must not mention policy concept: {leak}",
            )
        check(arm[1:] == transcript, f"profile {name} must preserve the transcript")
        check(
            "1 question" in arm[0]["content"],
            f"profile {name} must carry the question budget",
        )

    # Profiles must be distinct instructions, not duplicates of one template.
    rendered = {
        LENGTH_CONTROL_PROFILES[name].format(question_budget=1)
        for name in LENGTH_CONTROL_PROFILES
    }
    check(
        len(rendered) == len(LENGTH_CONTROL_PROFILES),
        "every profile must render a distinct instruction",
    )

    # Regression guard: negative constraints made replies LONGER, not shorter.
    # Measured on a 12-case Qwen2.5-0.5B pilot: mean characters were 97 for a
    # mild negative profile and 155 for the strictest, and one case reached 450
    # under an explicit "reply in one or two sentences". A small model
    # elaborates a stack of prohibitions. Profiles must stay positive targets.
    for name, template in LENGTH_CONTROL_PROFILES.items():
        lowered = template.lower()
        for marker in NEGATIVE_MARKERS:
            check(
                marker not in lowered,
                f"profile {name} must stay a positive target, found {marker!r}: {template!r}",
            )

    # An unknown profile is an error rather than a silent fallback.
    try:
        build_length_matched_messages(
            transcript, {"verbosity": "low", "question_budget": 1}, "nope"
        )
        FAILURES.append("unknown profile should raise")
    except ValueError:
        pass

    check(
        DEFAULT_LENGTH_PROFILE in LENGTH_CONTROL_PROFILES,
        "the default profile must exist",
    )

    # A non-low verbosity still gets a budget instruction rather than nothing.
    normal = build_length_matched_messages(
        transcript, {"verbosity": "normal", "question_budget": 0}, "answer_only"
    )
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

    # A sweep records every arm and states that selection happens on length.
    sweep = experiment_manifest(
        n_cases=12,
        cases_path=cases_path,
        affect_model=affect_model,
        subtext_model=subtext_model,
        model=FakeModel(),
        global_seed=42,
        allow_annotations=False,
        length_profiles=("answer_only", "direct", "one_sentence", "shortest_complete"),
    )
    check(len(sweep["arms"]) == 6, f"sweep should record six arms, got {sweep['arms']}")
    check(
        all(f"length_matched:{n}" in sweep["arms"] for n in ("answer_only", "direct", "one_sentence", "shortest_complete")),
        "every profile must appear as its own arm",
    )
    check(
        "length only" in sweep["length_profile_selection"],
        "manifest must state that profile selection uses length only",
    )
    check("sweep" in sweep["design"], "sweep design should be named")

    # An unknown profile must be rejected at manifest time too.
    try:
        experiment_manifest(
            n_cases=1,
            cases_path=cases_path,
            affect_model=affect_model,
            subtext_model=subtext_model,
            model=FakeModel(),
            global_seed=42,
            allow_annotations=False,
            length_profiles=("does_not_exist",),
        )
        FAILURES.append("unknown profile should be rejected by the manifest")
    except ValueError:
        pass

    shutil.rmtree(stub_dir, ignore_errors=True)
    print(json.dumps({"failures": FAILURES, "ok": not FAILURES}, indent=2))
    return 1 if FAILURES else 0


class FakeModel:
    model_id = "test-model"
    temperature = 0.0
    command = ["python", "server.py"]


if __name__ == "__main__":
    raise SystemExit(main())