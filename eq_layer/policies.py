from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
        """Three consecutive turns before this one that all moved upward.

        The window deliberately excludes the current turn. Snapshot
        classification cannot see this: one polite message after three bad
        ones reads as calm, and that is exactly when the register must not
        return to warm."""
        recent = self.history[-4:-1]
        return len(recent) >= 3 and all(s.escalation_delta > 0.05 for s in recent)

    @property
    def user_is_right(self) -> bool:
        return self.stance == "user_right"

    @property
    def user_is_wrong(self) -> bool:
        return self.stance == "user_wrong"

    @property
    def stance_unknown(self) -> bool:
        return self.stance == "unknown"


@dataclass(frozen=True)
class Policy:
    name: str
    register: Register
    summary: str
    preconditions: tuple[str, ...]
    avoid: tuple[str, ...] = ()
    example_opener: str = ""

    @property
    def specificity(self) -> int:
        """How many conditions must hold. Selection is not first-match:
        the most constrained applicable policy wins, so adding a policy can
        never silently change what a looser one does."""
        return len(self.preconditions)

    def applies(self, state: AffectState) -> bool:
        return all(PREDICATES[name](state) for name in self.preconditions)


PREDICATES = {
    "any": lambda s: True,
    "escalating": lambda s: s.escalating,
    "escalating_past_n": lambda s: s.escalating_past_n,
    "not_escalating": lambda s: not s.escalating,
    "not_escalating_past_n": lambda s: not s.escalating_past_n,
    "user_is_right": lambda s: s.user_is_right,
    "user_is_wrong": lambda s: s.user_is_wrong,
    "stance_unknown": lambda s: s.stance_unknown,
    "low_arousal": lambda s: s.arousal < 0.4,
    "high_arousal": lambda s: s.arousal >= 0.5,
    "high_arousal_run": lambda s: len(s.history) >= 2 and all(x.arousal >= 0.5 for x in s.history[-2:]),
}

SUBTEXTS = (
    "correction",
    "disclosure_request",
    "exhaustion",
    "resignation",
    "escalating",
    "demand",
    "challenge",
    "question",
    "statement",
)

for _subtext in SUBTEXTS:
    PREDICATES[f"subtext:{_subtext}"] = (lambda t: lambda s: s.subtext == t)(_subtext)

PREDICATES["subtext_set:drained"] = lambda s: s.subtext in {"exhaustion", "resignation"}


POLICIES: tuple[Policy, ...] = (
    Policy(
        name="mirror_specific",
        register=Register.MIRROR,
        summary="Name the concrete thing the user said, not a category of difficulty.",
        preconditions=("subtext:statement", "not_escalating", "not_escalating_past_n"),
        avoid=("That sounds really tough.", "I understand how you feel."),
        example_opener="Randevunuz üç gündür iki saat ileriye alınıyor, sonra da kırk dakikalık yol var.",
    ),
    Policy(
        name="validate_then_redirect",
        register=Register.VALIDATE_REDIRECT,
        summary="The user is right and heated: acknowledge, then move to one concrete step.",
        preconditions=("subtext:challenge", "user_is_right"),
        avoid=("Validating and stopping there.", "Apologising again."),
        example_opener="Rehber değişti, üstüne müdahale edildi. Sonraki adım: yeni metni karşılaştırıp farkı tek cümlede yazalım.",
    ),
    Policy(
        name="deflate_tension",
        register=Register.DEFLATE,
        summary="Stacked demands: lower the temperature, do not lower the seriousness.",
        preconditions=("subtext:demand",),
        avoid=("Jokes that minimise a real problem.", "Answering a different demand than the one asked."),
        example_opener="Üç şey soruyorsun, tek tek gideyim. Önce şu: değişen kısım hangi satır?",
    ),
    Policy(
        name="direct_no_padding",
        register=Register.DIRECT,
        summary="No warmth before the answer when the user asked a plain question.",
        preconditions=("subtext:question", "not_escalating"),
        avoid=("Opening with empathy before the answer.", "Restating the question."),
        example_opener="Hayır, o dosya senin durumunda çalışmaz.",
    ),
    Policy(
        name="ask_one_question",
        register=Register.ASK,
        summary="After a correction, ask exactly one question — the one that unblocks the next step.",
        preconditions=("subtext:correction",),
        avoid=("Three or more questions.", "Questions stacked into an interrogation."),
        example_opener="Haklısın, tarihi yanlış okumuşum. Doğru tarih yaklaşık kaçıncıydı?",
    ),
    Policy(
        name="repair_hold_position",
        register=Register.REPAIR,
        summary="Pushed repeatedly: keep the position. Neither capitulate nor argue.",
        preconditions=("subtext:challenge", "stance_unknown"),
        avoid=("Total capitulation.", "Defending why you were right.", "Re-explaining the same thing."),
        example_opener="Haklısın, yanlış okudum. Yine de önerim aynı çünkü gerekçesi değişmedi.",
    ),
    Policy(
        name="repair_interrogation",
        register=Register.REPAIR,
        summary="The same question three turns running is not curiosity. Stop restating.",
        preconditions=("subtext:escalating",),
        avoid=("A fourth restatement of the explanation.", "Asking what the user actually needs."),
        example_opener="Üç kez aynı şeyi sordun. Sebebi şu ve değişmiyor: geçersiz tarih.",
    ),
    Policy(
        name="hold",
        register=Register.HOLD,
        summary="Energy is spent rather than rising. Say less.",
        preconditions=("subtext_set:drained",),
        avoid=("Continuing to generate reassurance.", "Cheerfulness.",),
        example_opener="Tamam. Düşüneyim.",
    ),
    Policy(
        name="boundary",
        register=Register.BOUNDARY,
        summary="Decline without moralising about the request.",
        preconditions=("subtext:disclosure_request",),
        avoid=("Lecturing.", "Repeating the refusal after it was accepted."),
        example_opener="Kimseye söylemem. Sadece bu sohbette kalır.",
    ),
    Policy(
        name="hold_sustained_escalation",
        register=Register.HOLD,
        summary="Three rising turns then a calm one: the calm turn is not a reason to warm back up.",
        preconditions=("escalating_past_n", "not_escalating"),
        avoid=("Returning to a warm register because the last message was polite.",),
        example_opener="Buradayım.",
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
    overrides: dict[str, str] = field(default_factory=dict)
    default: str = "mirror_specific"

    def select(self, state: AffectState) -> Selection:
        forced = self.overrides.get(state.subtext)
        if forced:
            return Selection(policy=REGISTRY[forced], state=state, rationale=f"override:{forced}")

        applicable = [p for p in POLICIES if p.applies(state)]
        if not applicable:
            return Selection(policy=REGISTRY[self.default], state=state, rationale="default")

        best = max(p.specificity for p in applicable)
        winners = [p for p in applicable if p.specificity == best]
        return Selection(
            policy=winners[0],
            state=state,
            rationale=f"specificity={best} matched={','.join(p.name for p in applicable)}",
            runner_up=tuple(p.name for p in applicable if p is not winners[0]),
        )