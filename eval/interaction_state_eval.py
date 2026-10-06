"""Regression tests for repair lifecycle and interaction quality.

Usage: python eval/interaction_state_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.interaction import (  # noqa: E402
    infer_interaction_quality,
    infer_repair_state,
)


def assert_close(name: str, got: float, expected: float) -> None:
    if abs(got - expected) > 1e-9:
        raise AssertionError(f"{name}: expected {expected}, got {got}")


def main() -> int:
    # First-turn clarification is not user->assistant repair.
    first_turn = [{"role": "user", "content": "Demek istediğim sürüm numarası."}]
    repair = infer_repair_state(first_turn)
    if repair.active:
        raise AssertionError("First-turn clarification was incorrectly marked as repair.")
    quality = infer_interaction_quality(first_turn)
    assert_close("first-turn quality", quality.current, 1.0)

    # Explicit correction after an assistant turn opens a structural repair.
    correction = [
        {"role": "assistant", "content": "Tarihi soruyorsun."},
        {"role": "user", "content": "Hayır, onu demedim. Sürümü soruyorum."},
    ]
    repair = infer_repair_state(correction)
    if not repair.active:
        raise AssertionError("Explicit user correction did not open repair.")
    if repair.kind != "user_corrects_assistant":
        raise AssertionError(f"Wrong repair kind: {repair.kind}")
    if repair.target_turn_index != 0:
        raise AssertionError(f"Wrong repair target: {repair.target_turn_index}")
    if repair.repeated:
        raise AssertionError("Single repair was incorrectly marked repeated.")
    quality = infer_interaction_quality(correction)
    assert_close("single repair quality", quality.current, 0.7)
    assert_close("single repair delta", quality.delta, -0.3)
    if quality.unresolved_repair_count != 1:
        raise AssertionError("Active repair was not counted as unresolved.")

    if not repair.replacement_explicit:
        raise AssertionError("Explicit Turkish correction replacement was not extracted.")
    if "Sürümü soruyorum" not in repair.replacement_excerpt:
        raise AssertionError(
            f"Wrong Turkish replacement: {repair.replacement_excerpt!r}"
        )

    # Once the assistant replies, the old correction is no longer structurally
    # unresolved. This does not claim the reply actually fixed the problem.
    responded = [
        *correction,
        {"role": "assistant", "content": "Sürüm numarası 1.4."},
    ]
    repair = infer_repair_state(responded)
    if repair.active:
        raise AssertionError("Responded repair remained artificially active.")
    if repair.kind != "responded_repair":
        raise AssertionError(f"Expected responded_repair, got {repair.kind}")
    quality = infer_interaction_quality(responded)
    if quality.unresolved_repair_count != 0:
        raise AssertionError("Responded repair still counted as unresolved.")

    # Repeated user corrections are an interaction-failure signal independent
    # of emotional arousal.
    repeated = [
        {"role": "assistant", "content": "Tarihi soruyorsun."},
        {"role": "user", "content": "Hayır, onu demedim. Sürümü soruyorum."},
        {"role": "assistant", "content": "Yani tarihi tekrar soruyorsun."},
        {"role": "user", "content": "Ben onu demedim. Sürüm numarasını soruyorum."},
    ]
    repair = infer_repair_state(repeated)
    if not repair.active or not repair.repeated:
        raise AssertionError(f"Repeated repair was not detected: {repair}")
    if repair.recent_repair_count != 2:
        raise AssertionError(f"Expected 2 recent repairs, got {repair.recent_repair_count}")
    quality = infer_interaction_quality(repeated)
    assert_close("repeated repair quality", quality.current, 0.55)
    assert_close("repeated repair delta", quality.delta, -0.45)
    if quality.repeated_failure_count != 1:
        raise AssertionError("Repeated repair did not increment failure count.")

    english_correction = [
        {"role": "assistant", "content": "You want AMC Mercado 24."},
        {"role": "user", "content": "I said AMC Mountain 16."},
    ]
    if not infer_repair_state(english_correction).active:
        raise AssertionError("Explicit English correction was not detected.")

    english_repair = infer_repair_state(english_correction)
    if not english_repair.replacement_explicit:
        raise AssertionError("English 'I said' replacement was not extracted.")
    if english_repair.replacement_excerpt != "AMC Mountain 16":
        raise AssertionError(
            f"Wrong English replacement: {english_repair.replacement_excerpt!r}"
        )

    dolittle = infer_repair_state(
        [
            {"role": "assistant", "content": "Did you say Gretel and Hansel?"},
            {"role": "user", "content": "No, I said I want to watch Dolittle."},
        ]
    )
    if not dolittle.replacement_explicit or "Dolittle" not in dolittle.replacement_excerpt:
        raise AssertionError(f"Dolittle correction was not extracted: {dolittle}")

    harkins = infer_repair_state(
        [
            {"role": "assistant", "content": "AMC, okay."},
            {"role": "user", "content": "I meant Harkins."},
        ]
    )
    if not harkins.replacement_explicit or harkins.replacement_excerpt != "Harkins":
        raise AssertionError(f"Harkins correction was not extracted: {harkins}")

    missing_replacement = infer_repair_state(
        [
            {"role": "assistant", "content": "You said AMC Mercado 24."},
            {"role": "user", "content": "I didn't say that."},
        ]
    )
    if not missing_replacement.active:
        raise AssertionError("Correction without replacement was not detected.")
    if missing_replacement.replacement_explicit:
        raise AssertionError(
            f"Non-explicit correction invented a replacement: {missing_replacement}"
        )

    english_nonrepair = [
        {"role": "assistant", "content": "Genre means category."},
        {"role": "user", "content": "What do you mean by genre?"},
    ]
    if infer_repair_state(english_nonrepair).active:
        raise AssertionError("Ordinary English clarification was misclassified as repair.")

    # User self-correction must remain separate from correcting the assistant.
    self_correction = [
        {"role": "assistant", "content": "Tarih 14 Ekim."},
        {"role": "user", "content": "Haklısın, ben yanlış tarihi okumuşum."},
    ]
    repair = infer_repair_state(self_correction)
    if repair.active:
        raise AssertionError("User self-correction was treated as assistant repair.")

    print("interaction quality and repair state: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
