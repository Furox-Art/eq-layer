"""Regression tests for explicit session preference state.

Usage: python eval/session_preferences_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.preferences import infer_session_preferences  # noqa: E402


def main() -> int:
    transient = infer_session_preferences(
        [{"role": "user", "content": "Kısa cevap ver."}]
    )
    if transient.verbosity is not None:
        raise AssertionError(
            "One-turn brevity instruction was incorrectly persisted as a session preference."
        )

    turkish = infer_session_preferences(
        [
            {
                "role": "user",
                "content": "Bu sohbet boyunca benimle kısa ve net konuş. Tahmin etme.",
            }
        ]
    )
    if turkish.verbosity != "low":
        raise AssertionError(turkish)
    if turkish.directness != "high":
        raise AssertionError(turkish)
    if not turkish.no_guess:
        raise AssertionError(turkish)
    if turkish.source_turns != (0,):
        raise AssertionError(turkish.source_turns)

    overridden = infer_session_preferences(
        [
            {
                "role": "user",
                "content": "Bu sohbet boyunca kısa ve net konuş.",
            },
            {"role": "assistant", "content": "Tamam."},
            {
                "role": "user",
                "content": "Bundan sonra detaylı konuş.",
            },
        ]
    )
    if overridden.verbosity != "high":
        raise AssertionError(
            f"Later explicit session preference did not override verbosity: {overridden}"
        )
    if overridden.directness != "high":
        raise AssertionError(
            "Independent prior directness preference should remain active."
        )
    if overridden.source_turns != (0, 2):
        raise AssertionError(overridden.source_turns)

    english = infer_session_preferences(
        [
            {
                "role": "user",
                "content": (
                    "From now on, keep your responses concise and direct. "
                    "Do not guess."
                ),
            }
        ]
    )
    if (
        english.verbosity != "low"
        or english.directness != "high"
        or not english.no_guess
    ):
        raise AssertionError(english)

    emotional = infer_session_preferences(
        [
            {
                "role": "user",
                "content": "Şu an sinirliyim, kısa kes.",
            }
        ]
    )
    if emotional.verbosity is not None:
        raise AssertionError(
            "Momentary affect/style wording was incorrectly promoted to session state."
        )

    print("session preferences: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
