"""Verify policy tie resolution is explicit rather than declaration-ordered.

Usage: python eval/policy_selector_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import eq_layer.policies as policies  # noqa: E402


def state() -> policies.AffectState:
    return policies.AffectState(
        valence=0.0,
        arousal=0.2,
        escalation_delta=0.0,
        stance="unknown",
        subtext="statement",
    )


def main() -> int:
    original = policies.POLICIES
    try:
        first = policies.Policy(
            name="first",
            register=policies.Register.DIRECT,
            summary="first",
            preconditions=("any",),
            priority=0,
        )
        second = policies.Policy(
            name="second",
            register=policies.Register.DIRECT,
            summary="second",
            preconditions=("any",),
            priority=0,
        )
        policies.POLICIES = (first, second)

        try:
            policies.Selector().select(state())
        except ValueError as exc:
            if "Ambiguous policy selection" not in str(exc):
                raise
        else:
            raise AssertionError("Equal specificity/priority must raise ambiguity.")

        second = policies.Policy(
            name="second",
            register=policies.Register.DIRECT,
            summary="second",
            preconditions=("any",),
            priority=10,
        )
        policies.POLICIES = (first, second)
        selected = policies.Selector().select(state())
        if selected.policy.name != "second":
            raise AssertionError(
                f"Explicit priority did not win: {selected.policy.name}"
            )
    finally:
        policies.POLICIES = original

    print("policy selector tie handling: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
