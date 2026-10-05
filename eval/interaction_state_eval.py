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
