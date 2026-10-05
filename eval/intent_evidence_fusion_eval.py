"""Regression tests for dialogue-evidence fusion into intent belief.

Usage: python eval/intent_evidence_fusion_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.intent_belief import (  # noqa: E402
    IntentBelief,
    decide_intent_action,
    fuse_dialogue_evidence,
)


def main() -> int:
    raw = IntentBelief.from_mapping(
        {
            "action_request": 0.15,
            "status_check": 0.10,
            "explanation": 0.10,
            "question": 0.10,
            "statement": 0.30,
            "unknown": 0.25,
        }
    )
    raw_snapshot = raw.as_dict()
    raw_decision = decide_intent_action(raw)
    if raw_decision.action != "clarify":
        raise AssertionError(f"Expected diffuse raw belief to clarify: {raw_decision}")

    fused, evidence = fuse_dialogue_evidence(
        raw,
        text="I appreciate that. Things are better now.",
        dialogue_act="inform",
        act_confidence=0.90,
    )
    fused_decision = decide_intent_action(fused)
    if fused_decision.action != "statement":
        raise AssertionError(
            f"High-confidence inform evidence should support contextual response: "
            f"{fused_decision}"
        )
    if fused.probability("statement") <= raw.probability("statement"):
        raise AssertionError("Inform evidence did not increase statement posterior.")
    if raw.as_dict() != raw_snapshot:
        raise AssertionError("Fusion mutated the raw posterior in place.")
    if not any(item.startswith("dialogue_act:inform") for item in evidence):
        raise AssertionError(f"Dialogue-act evidence provenance missing: {evidence}")

    # Surface punctuation is independent evidence and must protect an explicit
    # question even when the dialogue-act model incorrectly says "inform".
    question_raw = IntentBelief.from_mapping(
        {
            "action_request": 0.10,
            "status_check": 0.15,
            "explanation": 0.15,
            "question": 0.20,
            "statement": 0.30,
            "unknown": 0.10,
        }
    )
    question_fused, question_evidence = fuse_dialogue_evidence(
        question_raw,
        text="Is that correct?",
        dialogue_act="inform",
        act_confidence=0.90,
    )
    if decide_intent_action(question_fused).action != "question":
        raise AssertionError(
            "Explicit question mark was overridden by incorrect dialogue-act evidence."
        )
    if "structure:question_mark" not in question_evidence:
        raise AssertionError("Question-mark structural evidence was not recorded.")

    directive_raw = IntentBelief.from_mapping(
        {
            "action_request": 0.30,
            "status_check": 0.10,
            "explanation": 0.10,
            "question": 0.10,
            "statement": 0.25,
            "unknown": 0.15,
        }
    )
    directive_fused, _ = fuse_dialogue_evidence(
        directive_raw,
        text="Please do it.",
        dialogue_act="directive",
        act_confidence=0.90,
    )
    if decide_intent_action(directive_fused).action != "action_request":
        raise AssertionError("Directive evidence did not support action routing.")

    print("intent dialogue-evidence fusion: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
