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
        if "condition=baseline" not in pair["baseline"]:
            raise AssertionError("Baseline did not use baseline condition.")
        if "condition=eq" not in pair["eq"]:
            raise AssertionError("EQ response did not use EQ condition.")
        if "messages=3" in pair["baseline"] and pair["id"] == "heldout-001":
            raise AssertionError("Baseline unexpectedly received the EQ system message.")
        if pair["id"] == "heldout-001" and "messages=3" not in pair["eq"]:
            raise AssertionError("EQ condition lost original transcript context.")

    if (
        pairs[0]["generation"]["pair_seed"]
        != stable_case_seed(17, "heldout-001")
    ):
        raise AssertionError("Per-case seed is not deterministic.")

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

    print("response experiment engine: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
