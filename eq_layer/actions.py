from __future__ import annotations

from dataclasses import asdict, dataclass

from .intent import IntentState
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
    """Inspectable response action compiled from intent and dialogue policy.

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


def _task_move(intent: IntentState | None, policy: Policy) -> str:
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


def _realization(policy: Policy, intent: IntentState | None) -> RealizationControls:
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

    repair_move = REPAIR_MOVES.get(policy.name, "none")
    question_budget = 1 if (
        repair_move in {"clarify_goal", "clarify_one"}
        or (intent is not None and intent.needs_clarification)
    ) else 0

    return RealizationControls(
        verbosity=verbosity,
        directness=directness,
        warmth=warmth,
        question_budget=question_budget,
        scope_limited="scope_limited" in constraints,
        no_guess=(
            "no_guess" in constraints
            or (intent is not None and intent.needs_clarification)
        ),
    )


def compose_action(
    policy: Policy,
    intent: IntentState | None = None,
) -> FactoredAction:
    """Compile one selected legacy policy into orthogonal response controls."""

    task_move = _task_move(intent, policy)
    if policy.name == "boundary":
        task_move = "set_boundary"

    return FactoredAction(
        task_move=task_move,
        social_move=SOCIAL_MOVES.get(policy.name, "neutral"),
        repair_move=REPAIR_MOVES.get(policy.name, "none"),
        realization=_realization(policy, intent),
        source_policy=policy.name,
    )
