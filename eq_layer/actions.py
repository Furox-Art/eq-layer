from __future__ import annotations

from dataclasses import asdict, dataclass

from .intent import IntentState
from .interaction import InteractionQualityState, RepairState
from .policies import Policy, Register


@dataclass(frozen=True)
class RealizationControls:
    """Surface-form controls kept separate from task/social/repair decisions."""

    verbosity: str = "normal"
    directness: str = "normal"
    warmth: str = "normal"
    question_budget: int = 0
    scope_limited: bool = False
    no_guess: bool = False


@dataclass(frozen=True)
class FactoredAction:
    """Inspectable response action compiled from intent and dialogue state.

    task_move is driven primarily by the user's explicit goal.
    social_move and repair_move are driven by conversational state/policy.
    This prevents affective adaptation from silently replacing the task.
    """

    task_move: str
    social_move: str
    repair_move: str
    realization: RealizationControls
    source_policy: str

    def to_dict(self) -> dict:
        return asdict(self)


def _task_move(
    intent: IntentState | None,
    policy: Policy,
    repair: RepairState | None = None,
) -> str:
    if repair is not None and repair.active:
        if repair.replacement_explicit:
            return "resume_prior_task_with_correction"
        return "clarify_correction"

    if intent is not None:
        if intent.needs_clarification:
            return "clarify_goal"
        return {
            "action_request": "execute_request",
            "status_check": "report_status",
            "explanation": "explain",
            "question": "answer_question",
            "statement": "respond_contextually",
            "unknown": "clarify_goal",
        }.get(intent.kind, "respond_contextually")

    return {
        "execute_request": "execute_request",
        "report_status": "report_status",
        "explain_request": "explain",
        "clarify_request": "clarify_goal",
        "direct_no_padding": "answer_question",
        "boundary": "set_boundary",
    }.get(policy.name, "respond_contextually")


SOCIAL_MOVES = {
    "mirror_specific": "mirror_specific",
    "validate_then_redirect": "validate_specific",
    "deflate_tension": "deflate_tension",
    "hold": "minimal_presence",
    "boundary": "boundary_neutral",
    "hold_sustained_escalation": "low_warmth",
}

REPAIR_MOVES = {
    "clarify_request": "clarify_goal",
    "ask_one_question": "clarify_one",
    "repair_hold_position": "hold_position",
    "repair_interrogation": "stop_restatement",
}


def _repair_move(policy: Policy, repair: RepairState | None) -> str:
    base = REPAIR_MOVES.get(policy.name, "none")
    if repair is None or not repair.active:
        return base
    if repair.replacement_explicit:
        if repair.repeated:
            return "stop_restatement_and_apply_correction"
        return "apply_correction"
    if repair.repeated:
        return "stop_restatement_and_repair"
    if base != "none":
        return base
    return "clarify_one"


def _realization(
    policy: Policy,
    intent: IntentState | None,
    *,
    repair_move: str,
    repair: RepairState | None = None,
    interaction_quality: InteractionQualityState | None = None,
) -> RealizationControls:
    by_register = {
        Register.MIRROR: ("normal", "normal", "normal"),
        Register.VALIDATE_REDIRECT: ("normal", "high", "normal"),
        Register.DIRECT: ("normal", "high", "low"),
        Register.ASK: ("low", "high", "normal"),
        Register.DEFLATE: ("low", "high", "low"),
        Register.REPAIR: ("low", "high", "low"),
        Register.HOLD: ("low", "normal", "low"),
        Register.BOUNDARY: ("low", "high", "low"),
    }
    verbosity, directness, warmth = by_register[policy.register]

    constraints = set(intent.constraints) if intent is not None else set()
    if "brief" in constraints:
        verbosity = "low"
    if "detailed" in constraints:
        verbosity = "high"
    if "direct" in constraints:
        directness = "high"

    # Interaction quality is not affect. Only a strong structural failure
    # signal changes realization, and it does so conservatively.
    if interaction_quality is not None and interaction_quality.current <= 0.55:
        verbosity = "low"
        directness = "high"
        warmth = "low"

    explicit_replacement = (
        repair is not None
        and repair.active
        and repair.replacement_explicit
    )
    repair_needs_clarification = (
        repair is not None
        and repair.active
        and not repair.replacement_explicit
    )
    question_budget = 1 if (
        explicit_replacement
        or repair_needs_clarification
        or repair_move in {"clarify_goal", "clarify_one"}
        or (
            intent is not None
            and intent.needs_clarification
            and not explicit_replacement
        )
    ) else 0

    no_guess = (
        "no_guess" in constraints
        or repair_needs_clarification
        or (
            intent is not None
            and intent.needs_clarification
            and not explicit_replacement
        )
    )

    # An active correction identifies a repair target, but does not prove the
    # user's factual stance. Do not promote it to user_right/user_wrong.
    if repair is not None and repair.active and repair_move == "clarify_one":
        no_guess = True

    return RealizationControls(
        verbosity=verbosity,
        directness=directness,
        warmth=warmth,
        question_budget=question_budget,
        scope_limited="scope_limited" in constraints,
        no_guess=no_guess,
    )


def compose_action(
    policy: Policy,
    intent: IntentState | None = None,
    *,
    repair: RepairState | None = None,
    interaction_quality: InteractionQualityState | None = None,
) -> FactoredAction:
    """Compile policy + intent + dialogue-health state into orthogonal controls."""

    task_move = _task_move(intent, policy, repair)
    if policy.name == "boundary":
        task_move = "set_boundary"

    repair_move = _repair_move(policy, repair)
    social_move = SOCIAL_MOVES.get(policy.name, "neutral")

    if (
        interaction_quality is not None
        and interaction_quality.current <= 0.55
        and social_move == "neutral"
    ):
        social_move = "low_warmth"

    return FactoredAction(
        task_move=task_move,
        social_move=social_move,
        repair_move=repair_move,
        realization=_realization(
            policy,
            intent,
            repair_move=repair_move,
            repair=repair,
            interaction_quality=interaction_quality,
        ),
        source_policy=policy.name,
    )
