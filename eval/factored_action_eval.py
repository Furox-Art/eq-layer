"""Regression tests for orthogonal factored response actions.

Usage: python eval/factored_action_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.actions import compose_action  # noqa: E402
from eq_layer.intent import IntentState  # noqa: E402
from eq_layer.interaction import InteractionQualityState, RepairState  # noqa: E402
from eq_layer.policies import REGISTRY  # noqa: E402
from eq_layer.steer import Steer  # noqa: E402


def intent(
    kind: str,
    *,
    clarify: bool = False,
    constraints: tuple[str, ...] = (),
) -> IntentState:
    return IntentState(
        kind=kind,
        canonical_request=f"test:{kind}",
        response_mode="clarify" if clarify else "direct",
        confidence=0.9,
        explicit=kind != "statement",
        needs_clarification=clarify,
        constraints=constraints,
        evidence=("factored_action_eval",),
    )


def main() -> int:
    # Every production policy must compile into an inspectable action.
    for name, policy in REGISTRY.items():
        action = compose_action(policy, intent("statement"))
        if action.source_policy != name:
            raise AssertionError(f"{name}: source policy was not preserved")

    # Social adaptation must not silently replace the user's task.
    action_intent = intent(
        "action_request",
        constraints=("brief", "direct", "no_guess"),
    )
    direct = compose_action(REGISTRY["execute_request"], action_intent)
    deflated = compose_action(REGISTRY["deflate_tension"], action_intent)

    if direct.task_move != "execute_request":
        raise AssertionError(f"Explicit action was lost: {direct}")
    if deflated.task_move != direct.task_move:
        raise AssertionError(
            "Changing social policy changed task_move; factorization is not orthogonal."
        )
    if deflated.social_move != "deflate_tension":
        raise AssertionError("Deflate policy did not alter only the social component.")
    if direct.realization.verbosity != "low":
        raise AssertionError("brief constraint did not control verbosity.")
    if direct.realization.directness != "high":
        raise AssertionError("direct constraint did not control directness.")
    if not direct.realization.no_guess:
        raise AssertionError("no_guess constraint was lost.")

    # Repair is an independent gate: preserve the task while asking one question.
    correction = compose_action(REGISTRY["ask_one_question"], action_intent)
    if correction.task_move != "execute_request":
        raise AssertionError("Correction handling erased the original task.")
    if correction.repair_move != "clarify_one":
        raise AssertionError("Correction did not compile to clarify_one repair.")
    if correction.realization.question_budget != 1:
        raise AssertionError("Correction repair must allow exactly one question.")

    # An explicit correction replacement outranks latest-turn intent ambiguity.
    explicit_repair = RepairState(
        active=True,
        kind="user_corrects_assistant",
        target_turn_index=0,
        correction_excerpt="No, I said I want to watch Dolittle.",
        replacement_excerpt="I want to watch Dolittle",
        replacement_explicit=True,
        confidence=1.0,
        evidence=("test",),
    )
    explicit_action = compose_action(
        REGISTRY["ask_one_question"],
        intent("unknown", clarify=True),
        repair=explicit_repair,
    )
    if explicit_action.task_move != "resume_prior_task_with_correction":
        raise AssertionError(
            f"Explicit repair replacement did not preserve task progress: {explicit_action}"
        )
    if explicit_action.repair_move != "apply_correction":
        raise AssertionError(
            f"Explicit repair was not applied directly: {explicit_action}"
        )
    if explicit_action.realization.question_budget != 1:
        raise AssertionError("Corrected task must retain at most one next-step question.")

    missing_repair = RepairState(
        active=True,
        kind="user_corrects_assistant",
        target_turn_index=0,
        correction_excerpt="I didn't say that.",
        replacement_explicit=False,
        confidence=1.0,
        evidence=("test",),
    )
    missing_action = compose_action(
        REGISTRY["ask_one_question"],
        intent("unknown", clarify=True),
        repair=missing_repair,
    )
    if missing_action.task_move != "clarify_correction":
        raise AssertionError(
            f"Missing replacement did not request correction clarification: {missing_action}"
        )
    if missing_action.repair_move != "clarify_one":
        raise AssertionError(
            f"Missing replacement did not compile to one clarification: {missing_action}"
        )

    # Intent-level ambiguity gets one targeted clarification even without a repair label.
    ambiguous = compose_action(
        REGISTRY["mirror_specific"],
        intent("unknown", clarify=True),
    )
    if ambiguous.task_move != "clarify_goal":
        raise AssertionError("Ambiguous intent did not compile to clarify_goal.")
    if ambiguous.realization.question_budget != 1:
        raise AssertionError("Clarification task has no question budget.")
    if not ambiguous.realization.no_guess:
        raise AssertionError("Clarification task must forbid guessing.")

    # Repeated repair changes repair/social realization, not the user's task.
    repeated_repair = RepairState(
        active=True,
        kind="user_corrects_assistant",
        target_turn_index=0,
        repeated=True,
        recent_repair_count=2,
        confidence=1.0,
        evidence=("test",),
    )
    low_quality = InteractionQualityState(
        current=0.55,
        delta=-0.45,
        repeated_failure_count=1,
        unresolved_repair_count=1,
        clarification_count=0,
        evidence=("test",),
    )
    degraded = compose_action(
        REGISTRY["execute_request"],
        action_intent,
        repair=repeated_repair,
        interaction_quality=low_quality,
    )
    if degraded.task_move != "execute_request":
        raise AssertionError("Repeated repair erased the explicit task.")
    if degraded.repair_move != "stop_restatement_and_repair":
        raise AssertionError(f"Repeated repair move missing: {degraded.repair_move}")
    if degraded.social_move != "low_warmth":
        raise AssertionError(f"Low interaction quality was ignored: {degraded.social_move}")
    if (
        degraded.realization.verbosity != "low"
        or degraded.realization.directness != "high"
        or degraded.realization.warmth != "low"
    ):
        raise AssertionError(f"Low-quality realization controls are wrong: {degraded}")

    # Boundary behavior is a task-level override, not merely a warm/cold style.
    boundary = compose_action(REGISTRY["boundary"], intent("statement"))
    if boundary.task_move != "set_boundary":
        raise AssertionError("Boundary did not compile to a task-level boundary move.")

    # Steering must expose all factors to the generator.
    steered = Steer.build(REGISTRY["ask_one_question"], action_intent)
    instruction = steered.system_instruction()
    required = (
        "task_move: execute_request",
        "social_move: neutral",
        "repair_move: clarify_one",
        "question_budget=1",
        "must not be replaced",
    )
    missing = [value for value in required if value not in instruction]
    if missing:
        raise AssertionError(f"Factored controls missing from steering: {missing}")

    repaired_steer = Steer.build(
        REGISTRY["ask_one_question"],
        intent("unknown", clarify=True),
        repair=explicit_repair,
    )
    repaired_instruction = repaired_steer.system_instruction()
    required_repair_text = (
        "task_move: resume_prior_task_with_correction",
        "repair_move: apply_correction",
        "Explicit correction replacement",
        "Do not ask the user to repeat or reconfirm",
    )
    missing_repair_text = [
        value for value in required_repair_text
        if value not in repaired_instruction
    ]
    if missing_repair_text:
        raise AssertionError(
            f"Replacement-aware repair steering is incomplete: {missing_repair_text}"
        )
    if "Do not guess the missing referent. Ask exactly one targeted question." in repaired_instruction:
        raise AssertionError(
            "Explicit replacement still triggered generic intent clarification."
        )

    print("factored action architecture: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
