from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class Register(Enum):
    MIRROR = "mirror"
    VALIDATE_REDIRECT = "validate_redirect"
    DIRECT = "direct"
    ASK = "ask"
    DEFLATE = "deflate"
    REPAIR = "repair"
    HOLD = "hold"
    BOUNDARY = "boundary"


@dataclass(frozen=True)
class AffectState:
    valence: float
    arousal: float
    escalation_delta: float
    stance: str
    subtext: str
    turn_index: int = 0
    history: tuple["AffectState", ...] = ()

    @property
    def escalating(self) -> bool:
        return self.escalation_delta > 0.15

    @property
    def escalating_past_n(self) -> bool:
        if not self.history:
            return False
        recent = self.history[-3:]
        return len(recent) >= 3 and all(s.escalation_delta > 0.05 for s in recent)

    @property
    def user_is_right(self) -> bool:
        return self.stance in {"user_right", "user_right_but_misdirected"}


@dataclass(frozen=True)
class Policy:
    name: str
    register: Register
    summary: str
    preconditions: tuple[str, ...]
    avoid: tuple[str, ...] = ()
    example_opener: str = ""

    def applies(self, state: AffectState) -> bool:
        checks = {
            "escalating": state.escalating,
            "escalating_past_n": state.escalating_past_n,
            "not_escalating": not state.escalating,
            "user_is_right": state.user_is_right,
            "user_is_wrong": not state.user_is_right,
            "low_arousal": state.arousal < 0.4,
            "high_arousal": state.arousal >= 0.4,
            "any": True,
        }
        return all(checks.get(name, False) for name in self.preconditions)


POLICIES: tuple[Policy, ...] = (
    Policy(
        name="mirror_specific",
        register=Register.MIRROR,
        summary="Name the concrete thing the user said, not a category of difficulty.",
        preconditions=("any",),
        avoid=("That sounds really tough.", "I understand how you feel."),
        example_opener="Randevunuz üç gündür iki saat ileriye alınıyor, sonra da 40 dakikalık yol var.",
    ),
    Policy(
        name="validate_then_redirect",
        register=Register.VALIDATE_REDIRECT,
        summary="Acknowledge the emotion, then move to the next concrete step.",
        preconditions=("escalating", "user_is_right"),
        avoid=("Only validating with no next step.",),
        example_opener="İki kez reddedilmiş olmak sinir bozucu. Sonraki adım: dilekçeyi yazıp aynı memura değil bir üst makama göndermek.",
    ),
    Policy(
        name="direct_no_padding",
        register=Register.DIRECT,
        summary="Skip warmth entirely when the user asked for a straight answer.",
        preconditions=("user_is_wrong", "low_arousal"),
        avoid=("Opening with empathy before the answer.",),
        example_opener="Hayır, o dosya senin durumunda çalışmaz. Sebebi şu.",
    ),
    Policy(
        name="ask_one_question",
        register=Register.ASK,
        summary="Ask exactly one question, the one that unblocks the next step.",
        preconditions=("not_escalating",),
        avoid=("Three or more questions in a row.", "Question stacking that reads as interrogation."),
        example_opener="Soruyu daraltayım: hatırladığın tarih mi var, yoksa sadece yaklaşık bir zaman mı?",
    ),
    Policy(
        name="deflate_tension",
        register=Register.DEFLATE,
        summary="Lower the temperature without dismissing the concern.",
        preconditions=("escalating", "high_arousal", "user_is_right"),
        avoid=("Jokes that minimise a real problem.",),
        example_opener="Sakinleş, bunu çözeriz. Önce şu tek şeyi ayıralım: sorun gecikme mi, iptal mi?",
    ),
    Policy(
        name="repair_hold_position",
        register=Register.REPAIR,
        summary="Correct the mistake, keep the position. Neither capitulate nor deflect.",
        preconditions=("any",),
        avoid=("Total capitulation after correction.", "Defensive explanation of why you were right."),
        example_opener="Haklısın, yanlış okudum. Yine de önerim aynı çünkü gerekçesi değişmedi.",
    ),
    Policy(
        name="hold",
        register=Register.HOLD,
        summary="Say less. Silence or a short line is the correct move.",
        preconditions=("escalating_past_n",),
        avoid=("Continuing to generate reassurance.",),
        example_opener="Tamam. Söylediğimi düşüneceğim.",
    ),
    Policy(
        name="boundary",
        register=Register.BOUNDARY,
        summary="Decline without moralising about the request.",
        preconditions=("any",),
        avoid=("Lecturing.", "Repeated refusal after it has been accepted."),
        example_opener="Bunu yapamam. Şunu yapabilirim.",
    ),
)

REGISTRY: dict[str, Policy] = {p.name: p for p in POLICIES}


@dataclass
class Selection:
    policy: Policy
    state: AffectState
    rationale: str = ""
    runner_up: tuple[str, ...] = ()


@dataclass
class Selector:
    order: tuple[str, ...] = tuple(p.name for p in POLICIES)
    overrides: dict[str, str] = field(default_factory=dict)

    def select(self, state: AffectState) -> Selection:
        forced = self.overrides.get(state.subtext) or self.overrides.get(state.stance)
        candidates = [REGISTRY[forced]] if forced else self._candidates(state)
        if not candidates:
            candidates = [REGISTRY["mirror_specific"]]
        return Selection(
            policy=candidates[0],
            state=state,
            rationale=f"matched {','.join(p.name for p in candidates)}",
            runner_up=tuple(p.name for p in candidates[1:]),
        )

    def _candidates(self, state: AffectState) -> list[Policy]:
        return [REGISTRY[n] for n in self.order if REGISTRY[n].applies(state)]