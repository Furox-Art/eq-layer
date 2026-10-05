from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from .policies import AffectState
from .trained_subtext import SubtextDecision, TrainedSubtext


@dataclass(frozen=True)
class StanceDecision:
    label: str
    confidence: float
    evidence: tuple[str, ...] = ()


class StanceResolver(Protocol):
    def resolve(
        self,
        messages: list[dict],
        annotated: dict | None = None,
    ) -> StanceDecision: ...


@dataclass
class AnnotationStanceResolver:
    """Only emits right/wrong when an external verifier or annotation supplied it."""

    def resolve(
        self,
        messages: list[dict],
        annotated: dict | None = None,
    ) -> StanceDecision:
        annotated = annotated or {}
        stance = str(annotated.get("stance", "unknown"))
        if stance not in {"user_right", "user_wrong", "unknown"}:
            raise ValueError(f"Unsupported stance: {stance}")
        if stance == "unknown":
            return StanceDecision(
                label="unknown",
                confidence=0.0,
                evidence=("no_external_verification",),
            )
        return StanceDecision(
            label=stance,
            confidence=1.0,
            evidence=("external_annotation",),
        )


@dataclass(frozen=True)
class TrackingResult:
    state: AffectState
    subtext: SubtextDecision
    stance: StanceDecision


@dataclass
class ConversationTracker:
    """Compose learned V/A state with learned dialogue signals and safe stance."""

    affect: object
    subtext: TrainedSubtext
    stance: StanceResolver = AnnotationStanceResolver()

    def infer(
        self,
        messages: list[dict],
        turn_index: int = 0,
        annotated: dict | None = None,
    ) -> TrackingResult:
        annotated = annotated or {}
        base = self.affect.infer(
            messages,
            turn_index=turn_index,
            annotated={k: v for k, v in annotated.items() if k != "subtext"},
        )
        subtext = self.subtext.infer(messages, base, annotated=annotated)
        stance = self.stance.resolve(messages, annotated=annotated)
        state = replace(
            base,
            subtext=subtext.label,
            stance=stance.label,
        )
        return TrackingResult(state=state, subtext=subtext, stance=stance)
