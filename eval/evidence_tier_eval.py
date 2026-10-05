"""Verify that EQ-Layer never collapses evidence provenance into one label.

Usage: python eval/evidence_tier_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.policies import AffectState  # noqa: E402
from eq_layer.tracker import AnnotationStanceResolver  # noqa: E402
from eq_layer.trained_subtext import DialogueSignals, TrainedSubtext  # noqa: E402


class FakeSubtext(TrainedSubtext):
    def __init__(self, signals: DialogueSignals):
        super().__init__()
        self._signals = signals

    def predict_signals(self, text: str) -> DialogueSignals:
        return self._signals


def affect(valence: float = 0.0, arousal: float = 0.2) -> AffectState:
    return AffectState(
        valence=valence,
        arousal=arousal,
        escalation_delta=0.0,
        stance="unknown",
        subtext="",
        history=(),
    )


def check(name: str, got: str, expected: str) -> None:
    if got != expected:
        raise AssertionError(f"{name}: expected {expected}, got {got}")


def main() -> int:
    neutral_question = DialogueSignals("question", "neutral", 0.9, 0.9)
    angry_question = DialogueSignals("question", "anger", 0.9, 0.9)
    sad_inform = DialogueSignals("inform", "sadness", 0.9, 0.9)

    annotated = FakeSubtext(neutral_question).infer(
        [{"role": "user", "content": "anything"}],
        affect(),
        annotated={"subtext": "exhaustion"},
    )
    check("annotation", annotated.evidence_level, "verified")

    correction = FakeSubtext(neutral_question).infer(
        [{"role": "user", "content": "Hayır, onu demedim. Sürümü soruyorum."}],
        affect(),
    )
    check("correction label", correction.label, "correction")
    check("correction evidence", correction.evidence_level, "structural")

    self_correction = FakeSubtext(neutral_question).infer(
        [{"role": "user", "content": "Haklısın, ben yanlış tarihi okumuşum."}],
        affect(),
    )
    if self_correction.label == "correction":
        raise AssertionError("User self-correction must not be treated as correcting the assistant.")

    question = FakeSubtext(neutral_question).infer(
        [{"role": "user", "content": "Which file is current?"}],
        affect(),
    )
    check("question", question.evidence_level, "direct_learned")

    challenge = FakeSubtext(angry_question).infer(
        [{"role": "user", "content": "Why are you doing this?"}],
        affect(valence=-0.3, arousal=0.7),
    )
    check("challenge", challenge.label, "challenge")
    check("challenge evidence", challenge.evidence_level, "derived")

    exhausted = FakeSubtext(sad_inform).infer(
        [{"role": "user", "content": "I cannot keep doing this."}],
        affect(valence=-0.4, arousal=0.25),
    )
    check("exhaustion", exhausted.label, "exhaustion")
    check("exhaustion evidence", exhausted.evidence_level, "derived")

    resolver = AnnotationStanceResolver()
    unknown = resolver.resolve([{"role": "user", "content": "No."}])
    verified = resolver.resolve(
        [{"role": "user", "content": "No."}],
        annotated={"stance": "user_right"},
    )
    check("stance unknown", unknown.evidence_level, "unknown")
    check("stance verified", verified.evidence_level, "verified")

    print("evidence tiers: 9/9")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
