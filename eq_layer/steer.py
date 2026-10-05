"""Measure a response against a case.

The metrics are deliberately narrow. There is no standard EQ benchmark, and a
scorer that rewards warmth will reward a model that says nothing in
particular — so specificity is the primary signal and genericness is a
penalty rather than a component.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .actions import FactoredAction, compose_action
from .intent import IntentState
from .interaction import InteractionQualityState, RepairState
from .policies import Policy, Register

# Cheap under human raters. A model that is not penalised for these will
# produce them, because they are the highest-scoring way to sound caring while
# saying nothing.
GENERIC_PHRASES = (
    "i understand how you feel",
    "that sounds really tough",
    "that must be frustrating",
    "i'm sorry for your frustration",
    "that sounds incredibly difficult",
    "your feelings are valid",
    "i hear you",
    "that's completely understandable",
    "anlıyorum",
    "haklısın, çok zor",
    "bunu anlıyorum",
)


@dataclass
class Steer:
    policy: Policy
    intent: IntentState | None = None
    action: FactoredAction | None = None
    repair: RepairState | None = None
    interaction_quality: InteractionQualityState | None = None
    prefix: str = ""
    logit_bias: dict[int, float] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        policy: Policy,
        intent: IntentState | None = None,
        *,
        repair: RepairState | None = None,
        interaction_quality: InteractionQualityState | None = None,
    ) -> "Steer":
        return cls(
            policy=policy,
            intent=intent,
            repair=repair,
            interaction_quality=interaction_quality,
            action=compose_action(
                policy,
                intent,
                repair=repair,
                interaction_quality=interaction_quality,
            ),
            prefix=f"[{policy.register.value}] ",
        )

    def system_instruction(self) -> str:
        """Return steering instructions without discarding conversation history."""
        avoid = "; ".join(self.policy.avoid)
        intent_block = ""
        if self.intent is not None:
            constraints = ", ".join(self.intent.constraints) or "none"
            intent_block = (
                f"\nUser intent: {self.intent.canonical_request}"
                f"\nIntent kind: {self.intent.kind}; response mode: {self.intent.response_mode}; "
                f"confidence: {self.intent.confidence:.2f}; constraints: {constraints}."
            )
            if self.intent.needs_clarification:
                intent_block += "\nDo not guess the missing referent. Ask exactly one targeted question."

        action = self.action or compose_action(
            self.policy,
            self.intent,
            repair=self.repair,
            interaction_quality=self.interaction_quality,
        )
        controls = action.realization
        action_block = (
            "\nControl action:"
            f"\n- task_move: {action.task_move}"
            f"\n- social_move: {action.social_move}"
            f"\n- repair_move: {action.repair_move}"
            "\n- realization: "
            f"verbosity={controls.verbosity}, "
            f"directness={controls.directness}, "
            f"warmth={controls.warmth}, "
            f"question_budget={controls.question_budget}, "
            f"scope_limited={str(controls.scope_limited).lower()}, "
            f"no_guess={str(controls.no_guess).lower()}."
            "\nThe task_move represents the user's goal and must not be replaced "
            "by affective/social adaptation. social_move may shape interpersonal "
            "delivery only. If repair_move is active and task completion would "
            "require guessing, perform the repair first; otherwise preserve task "
            "progress. Never exceed question_budget."
        )

        dialogue_state_block = ""
        if self.repair is not None:
            dialogue_state_block += (
                "\nRepair state:"
                f" active={str(self.repair.active).lower()},"
                f" kind={self.repair.kind},"
                f" repeated={str(self.repair.repeated).lower()},"
                f" target_turn={self.repair.target_turn_index}."
                "\nA repair signal identifies conversational misalignment only; "
                "it does not establish that either side is factually correct."
            )
        if self.interaction_quality is not None:
            dialogue_state_block += (
                "\nInteraction quality:"
                f" current={self.interaction_quality.current:.2f},"
                f" delta={self.interaction_quality.delta:.2f},"
                f" repeated_failure_count={self.interaction_quality.repeated_failure_count},"
                f" unresolved_repair_count={self.interaction_quality.unresolved_repair_count},"
                f" clarification_count={self.interaction_quality.clarification_count}."
                "\nInteraction quality measures conversation health, not the user's emotion."
            )

        return (
            f"Respond using the {self.policy.name} policy "
            f"({self.policy.register.value}). {self.policy.summary} "
            f"Avoid: {avoid}.{intent_block}{action_block}{dialogue_state_block}"
        )

    def apply_to_prompt(self, user_message: str) -> str:
        return f"{self.system_instruction()}\n\nUser: {user_message}"


def specificity_score(response: str, case: dict) -> float:
    """Fraction of the case's expected concepts the response actually names.

    This is the metric that fails the most when the scorer is loose, so it is
    the one to improve first if these numbers look good for the wrong reasons.
    """
    expected = [w.lower() for w in case.get("expected_concepts", [])]
    if not expected:
        return 0.0
    low = response.lower()
    return round(sum(1 for w in expected if w in low) / len(expected), 3)


def genericness_score(response: str) -> float:
    low = response.lower()
    hits = sum(1 for p in GENERIC_PHRASES if p in low)
    return round(min(1.0, hits / 2), 3)


def escalation_latency(selected_registers: list[str]) -> int | None:
    """Turns before the register changed off mirror. None means it never did."""
    for idx, reg in enumerate(selected_registers):
        if reg != Register.MIRROR.value:
            return idx
    return None


def score_case(
    response: str,
    case: dict,
    selected_registers: list[str] | None = None,
) -> dict:
    specificity = specificity_score(response, case)
    genericness = genericness_score(response)
    latency = escalation_latency(selected_registers or [])
    latency_penalty = 0.0 if latency is None else min(0.2, 0.05 * latency)
    return {
        "case_id": case.get("id"),
        "specificity": specificity,
        "genericness": genericness,
        "escalation_latency": latency,
        "composite": round(max(0.0, specificity * (1.0 - genericness) - latency_penalty), 3),
    }


def aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {"n": 0}

    def mean(key: str) -> float:
        vals = [s[key] for s in scores if s.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    return {
        "n": len(scores),
        "specificity": mean("specificity"),
        "genericness": mean("genericness"),
        "composite": mean("composite"),
        "unresolved_escalations": sum(1 for s in scores if s["escalation_latency"] is None),
    }