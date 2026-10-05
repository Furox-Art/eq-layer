from __future__ import annotations

import re
from dataclasses import dataclass, field

from .policies import Policy, Register

GENERIC_PHRASES = (
    "i understand how you feel",
    "that sounds really tough",
    "that must be frustrating",
    "i'm sorry for your frustration",
    "that sounds incredibly difficult",
    "your feelings are valid",
    "i hear you",
    "that's completely understandable",
)

SPANISH_ESCALATION_HINT = re.compile(r"[!?]{2,}")


@dataclass
class Steer:
    policy: Policy
    prefix: str = ""
    logit_bias: dict[int, float] = field(default_factory=dict)

    @classmethod
    def build(cls, policy: Policy) -> "Steer":
        return cls(policy=policy, prefix=f"[{policy.register.value}] ")

    def apply_to_prompt(self, user_message: str) -> str:
        instruction = (
            f"Respond using the {self.policy.name} policy "
            f"({self.policy.register.value}). {self.policy.summary} "
            f"Avoid: {'; '.join(self.policy.avoid)}."
        )
        return f"{instruction}\n\nUser: {user_message}"


def specificity_score(response: str, case: dict) -> float:
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
    for idx, reg in enumerate(selected_registers):
        if reg != Register.MIRROR.value:
            return idx
    return None


def score_case(response: str, case: dict, selected_registers: list[str] | None = None) -> dict:
    specificity = specificity_score(response, case)
    genericness = genericness_score(response)
    latency = escalation_latency(selected_registers or [])
    latency_penalty = 0.0 if latency is None else min(0.2, 0.05 * latency)
    composite = round(
        max(0.0, specificity * (1.0 - genericness) - latency_penalty),
        3,
    )
    return {
        "case_id": case.get("id"),
        "specificity": specificity,
        "genericness": genericness,
        "escalation_latency": latency,
        "composite": composite,
    }


def aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {"n": 0}
    def mean(key: str) -> float:
        vals = [s[key] for s in scores if s[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0
    return {
        "n": len(scores),
        "specificity": mean("specificity"),
        "genericness": mean("genericness"),
        "composite": mean("composite"),
        "unresolved_escalations": sum(1 for s in scores if s["escalation_latency"] is None),
    }