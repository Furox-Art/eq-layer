"""Regression tests for conservative short-reference resolution.

Usage: python eval/dialogue_reference_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.dialogue_reference import infer_dialogue_reference  # noqa: E402


def main() -> int:
    first_turn = [{"role": "user", "content": "devam et"}]
    reference = infer_dialogue_reference(first_turn)
    if not reference.active or reference.resolved:
        raise AssertionError(f"First-turn continuation guessed a target: {reference}")
    if not reference.requires_clarification:
        raise AssertionError("Unanchored first-turn continuation did not require clarification.")

    offer = [
        {"role": "user", "content": "Sensitivity bitince ne yapacağız?"},
        {
            "role": "assistant",
            "content": "İstersen sıradaki adımda QUD resolver ekleyebilirim.",
        },
        {"role": "user", "content": "yap"},
    ]
    reference = infer_dialogue_reference(offer)
    if not reference.resolved or reference.target_turn_index != 1:
        raise AssertionError(f"Immediate assistant offer was not resolved: {reference}")
    if reference.form != "execute_reference":
        raise AssertionError(f"Wrong execute-reference form: {reference.form}")

    continuation = [
        {"role": "user", "content": "Bu analizi çalıştır."},
        {"role": "assistant", "content": "İlk 12 case bitti."},
        {"role": "user", "content": "devam et"},
    ]
    reference = infer_dialogue_reference(continuation)
    if not reference.resolved or reference.target_turn_index != 0:
        raise AssertionError(f"Prior user goal was not resumed safely: {reference}")

    blocked = [
        {"role": "user", "content": "Analizi yap."},
        {"role": "assistant", "content": "Hangi veri setini kullanayım?"},
        {"role": "user", "content": "devam et"},
    ]
    reference = infer_dialogue_reference(blocked)
    if reference.resolved:
        raise AssertionError(
            "Continuation incorrectly bypassed an unresolved assistant question."
        )
    if not reference.requires_clarification:
        raise AssertionError("Blocked continuation did not request clarification.")

    revision = [
        {"role": "user", "content": "Metni yaz."},
        {"role": "assistant", "content": "Draft text."},
        {"role": "user", "content": "onu düzelt"},
    ]
    reference = infer_dialogue_reference(revision)
    if not reference.resolved or reference.target_turn_index != 1:
        raise AssertionError(f"Revision did not target prior assistant output: {reference}")
    if reference.form != "revise_reference":
        raise AssertionError(f"Wrong revision form: {reference.form}")

    stale_execute = [
        {"role": "assistant", "content": "Sonuç 4."},
        {"role": "user", "content": "bunu yap"},
    ]
    reference = infer_dialogue_reference(stale_execute)
    if reference.resolved:
        raise AssertionError("Deictic execute guessed without an assistant offer.")
    if not reference.requires_clarification:
        raise AssertionError("Ambiguous deictic execute did not require clarification.")

    ordinary = [
        {"role": "assistant", "content": "Sonuç 4."},
        {"role": "user", "content": "bunu neden yaptın?"},
    ]
    reference = infer_dialogue_reference(ordinary)
    if reference.active:
        raise AssertionError("Ordinary question was misclassified as short reference.")

    print("dialogue reference state: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
