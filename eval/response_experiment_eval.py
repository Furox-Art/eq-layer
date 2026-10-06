"""Engineering smoke test for the same-model response experiment engine."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.response_experiment import (  # noqa: E402
    CommandModel,
    experiment_manifest,
    generate_pairs,
    stable_case_seed,
    validate_experiment_cases,
)


class AuditFixModel:
    model_id = "audit-fix-model"
    temperature = 0.0

    def __init__(self):
        self.calls = []

    def generate(self, messages, *, seed, condition):
        self.calls.append((condition, seed, list(messages)))
        has_retry = any(
            message.get("role") == "system"
            and "previous draft violated eq-layer structural control contracts"
            in str(message.get("content", "")).lower()
            for message in messages
        )
        if has_retry:
            return "Which version number do you mean?"
        if condition == "baseline":
            return "Baseline response."
        return "I need more information before I can answer."

    def close(self):
        return None


class ClarifyPipeline:
    def steer_messages(self, messages, *, annotated=None):
        return (
            [
                {
                    "role": "system",
                    "content": "Ask one targeted clarification question.",
                },
                *messages,
            ],
            {
                "policy": "clarify_request",
                "register": "ask",
                "intent_kind": "unknown",
                "intent_confidence": 0.4,
                "factored_action": {
                    "task_move": "clarify_goal",
                    "social_move": "neutral",
                    "repair_move": "clarify_one",
                    "realization": {
                        "verbosity": "low",
                        "directness": "high",
                        "warmth": "normal",
                        "question_budget": 1,
                        "scope_limited": False,
                        "no_guess": True,
                    },
                },
                "subtext": "plain",
                "subtext_evidence_level": "fallback",
                "stance": "unknown",
                "stance_evidence_level": "unknown",
                "repair": {
                    "active": False,
                    "kind": "none",
                },
                "selection_rationale": "audit-regeneration-smoke",
            },
        )


class FakePipeline:
    def steer_messages(self, messages, *, annotated=None):
        return (
            [
                {
                    "role": "system",
                    "content": "EQ control only; preserve the original transcript.",
                },
                *messages,
            ],
            {
                "policy": "fake_policy",
                "register": "direct",
                "intent_kind": "question",
                "intent_confidence": 0.9,
                "subtext": "question",
                "subtext_evidence_level": "direct_learned",
                "stance": "unknown",
                "stance_evidence_level": "unknown",
                "selection_rationale": "smoke-test",
            },
        )


def main() -> int:
    command = f"{sys.executable} {Path(__file__).with_name('fake_model_command.py')}"
    model = CommandModel.from_shell(
        command,
        model_id="same-test-model",
        temperature=0.0,
        timeout_seconds=30,
    )
    cases = [
        {
            "id": "heldout-001",
            "transcript": [
                {"role": "assistant", "content": "Previous context."},
                {"role": "user", "content": "Status?"},
            ],
        },
        {
            "id": "heldout-002",
            "transcript": [
                {"role": "user", "content": "Explain the failure."},
            ],
        },
    ]

    pairs = generate_pairs(
        cases,
        model=model,
        pipeline=FakePipeline(),
        seed=17,
        allow_annotations=False,
    )

    if len(pairs) != 2:
        raise AssertionError(f"Expected two pairs, got {len(pairs)}")

    for pair in pairs:
        generation = pair["generation"]
        if generation["model_id"] != "same-test-model":
            raise AssertionError("Model id changed between conditions.")
        if generation["same_model_both_arms"] is not True:
            raise AssertionError("Same-model invariant not recorded.")
        if generation["annotations_used"]:
            raise AssertionError("Annotations unexpectedly entered the default experiment.")
        if sorted(generation["order"]) != ["baseline", "eq"]:
            raise AssertionError(f"Both arms were not generated: {generation['order']}")
        if "system=0" not in pair["baseline"]:
            raise AssertionError("Baseline unexpectedly received a system steering message.")
        if "system=1" not in pair["eq"]:
            raise AssertionError("EQ response did not receive its system steering message.")
        if "messages=3" in pair["baseline"] and pair["id"] == "heldout-001":
            raise AssertionError("Baseline unexpectedly received the EQ system message.")
        if pair["id"] == "heldout-001" and "messages=3" not in pair["eq"]:
            raise AssertionError("EQ condition lost original transcript context.")
        audit = generation.get("response_audit")
        if not isinstance(audit, dict):
            raise AssertionError("Passive response audit metadata is missing.")
        if "passed" not in audit or "issues" not in audit:
            raise AssertionError(f"Malformed response audit metadata: {audit}")
        if generation["auditor_regenerations_allowed"] != 0:
            raise AssertionError("Passive mode unexpectedly allowed regeneration.")
        if generation["audit_regeneration_count"] != 0:
            raise AssertionError("Passive mode unexpectedly regenerated a response.")
        if "eq_initial" in pair:
            raise AssertionError("Passive mode should not emit an alternate EQ draft.")

    if (
        pairs[0]["generation"]["pair_seed"]
        != stable_case_seed(17, "heldout-001")
    ):
        raise AssertionError("Per-case seed is not deterministic.")

    retry_model = AuditFixModel()
    retry_pairs = generate_pairs(
        [
            {
                "id": "retry-001",
                "transcript": [
                    {"role": "user", "content": "Which one?"}
                ],
            }
        ],
        model=retry_model,
        pipeline=ClarifyPipeline(),
        seed=23,
        auditor_regenerations=1,
    )
    retry_pair = retry_pairs[0]
    retry_generation = retry_pair["generation"]
    if retry_generation["audit_regeneration_count"] != 1:
        raise AssertionError(f"Expected one audit regeneration: {retry_generation}")
    if retry_generation["response_audit_initial"]["passed"]:
        raise AssertionError("Initial failing draft unexpectedly passed audit.")
    if not retry_generation["response_audit"]["passed"]:
        raise AssertionError(f"Regenerated response still failed audit: {retry_pair}")
    if "eq_initial" not in retry_pair:
        raise AssertionError("Regeneration provenance did not retain the initial EQ draft.")
    if retry_pair["eq_initial"] == retry_pair["eq"]:
        raise AssertionError("Regeneration did not replace the failing EQ draft.")
    if "?" not in retry_pair["eq"]:
        raise AssertionError("Regeneration did not produce the required clarification question.")

    pilot_preflight = validate_experiment_cases(cases, final=False)
    if not pilot_preflight["ok"]:
        raise AssertionError(f"Pilot preflight unexpectedly failed: {pilot_preflight}")

    leaked = [
        {
            "id": "leaked-001",
            "transcript": [{"role": "user", "content": "Status?"}],
            "expected_policy": "report_status",
        }
    ]
    leaked_report = validate_experiment_cases(leaked, final=True)
    if leaked_report["ok"] or not any(
        "development/gold fields" in error
        for error in leaked_report["errors"]
    ):
        raise AssertionError("Final preflight did not reject gold/development fields.")

    final_cases = [
        {
            "id": f"final-{index:03d}",
            "transcript": [{"role": "user", "content": f"Question {index}?"}],
        }
        for index in range(120)
    ]
    final_report = validate_experiment_cases(final_cases, final=True)
    if not final_report["ok"]:
        raise AssertionError(f"Valid final preflight failed: {final_report}")

    with tempfile.TemporaryDirectory(prefix="eq-layer-ab-smoke-") as tmp:
        tmp_path = Path(tmp)
        cases_path = tmp_path / "cases.jsonl"
        affect_path = tmp_path / "affect.bin"
        subtext_path = tmp_path / "subtext.bin"

        cases_path.write_text(
            '{"id":"heldout-001","transcript":[{"role":"user","content":"Status?"}]}\n',
            encoding="utf-8",
        )
        affect_path.write_bytes(b"affect-model")
        subtext_path.write_bytes(b"subtext-model")

        manifest = experiment_manifest(
            model=model,
            affect_model=affect_path,
            subtext_model=subtext_path,
            cases_path=cases_path,
            global_seed=17,
            n_cases=1,
            allow_annotations=False,
            git_commit="deadbeef",
        )

        if "model_command" in manifest:
            raise AssertionError("Raw model command must not be stored in the manifest.")
        if not manifest.get("model_command_argv_sha256"):
            raise AssertionError("Model command fingerprint missing.")
        if manifest["annotations_used"]:
            raise AssertionError("Manifest says annotations were used.")
        if manifest.get("response_auditor_mode") != "passive-metadata-only":
            raise AssertionError("Manifest did not preserve passive auditor mode.")

        retry_manifest = experiment_manifest(
            model=model,
            affect_model=affect_path,
            subtext_model=subtext_path,
            cases_path=cases_path,
            global_seed=17,
            n_cases=1,
            allow_annotations=False,
            git_commit="deadbeef",
            auditor_regenerations=1,
        )
        if retry_manifest.get("response_auditor_mode") != "one-shot-regeneration":
            raise AssertionError("One-shot auditor mode was not recorded in the manifest.")

    print("response experiment engine: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
