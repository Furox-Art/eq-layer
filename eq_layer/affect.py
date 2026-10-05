from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Protocol

from .policies import AffectState


class AffectAdapter(Protocol):
    name: str

    def infer(
        self,
        messages: list[dict],
        turn_index: int = 0,
        annotated: dict | None = None,
    ) -> AffectState: ...


# --- detectors -----------------------------------------------------------
# Keyword sets are a stand-in for a trained classifier. They are deliberately
# small and readable: when one of them is wrong you can see why without a
# gradient.

ESCALATION_MARKERS = (
    "neden",
    "sen de",
    "saçmalama",
    "yine",
    "kaç kez",
    "kaç defa",
    "anlamıyorsun",
    "sinir",
    "rezalet",
)

SOFTENERS = (
    "galiba",
    "aslında",
    "belki",
    "biraz",
    "emin değilim",
    "kısmen",
)

CORRECTION_MARKERS = (
    "hayır, onu demedim",
    "hayır onu demedim",
    "öyle değil",
    "yanlış anladın",
    "yanlış anladınız",
    "onu sormadım",
    "ben onu demedim",
    "demek istediğim",
    "kastettiğim",
    "that's not what i meant",
    "you misunderstood",
    "no, i meant",
    "i didn't say that",
)

DISCLOSURE_MARKERS = (
    "kimseye anlatma",
    "kimseye söyleme",
    "sır",
    "gizli tut",
    "benden başka kimse",
)

# A user opening consecutive turns with a wh-word is asking the same thing
# again. That is interrogation, not curiosity — and the register has to change.
QUESTION_OPENERS = ("ne", "neden", "niçin", "nasıl", "nedir", "hangi", "kaç", "mı", "mi")


@dataclass
class HeuristicAffect:
    name: str = "heuristic"

    # --- arousal ----------------------------------------------------------

    def _arousal(self, text: str) -> float:
        if not text:
            return 0.0
        low = text.lower().strip()
        hits = sum(1 for m in ESCALATION_MARKERS if m in low)
        softens = sum(1 for s in SOFTENERS if s in low)
        score = min(1.0, 0.25 + hits * 0.22 - softens * 0.12)
        if "?" in low or "!" in low:
            score = min(1.0, score + 0.1)
        return round(score, 3)

    def _has(self, text: str, markers: Iterable[str]) -> bool:
        low = text.lower()
        return any(m in low for m in markers)

    # --- structural signals ----------------------------------------------

    def _recent_user_turns(self, messages: list[dict], n: int = 4) -> list[str]:
        turns = [m["content"] for m in messages if m.get("role") == "user"]
        return turns[-n:]

    def _escalating(self, user_turns: list[str]) -> bool:
        """Three or more consecutive turns opening with a wh-word."""
        if len(user_turns) < 3:
            return False
        return all(
            self._first_word(t) in QUESTION_OPENERS for t in user_turns[-3:] if t.strip()
        )

    def _first_word(self, text: str) -> str:
        """First token with punctuation stripped. 'Neden?' must match 'neden',
        otherwise a bare one-word question — the shape this is built to
        detect — silently fails."""
        token = text.lower().strip().split()[0]
        return token.strip("?!.,;:")

    def _demand_pattern(self, user_turns: list[str]) -> bool:
        """Short imperative demands, stacked. 'Anlat biçim!' repeated."""
        recent = [t for t in user_turns[-3:] if t.strip()]
        if len(recent) < 3:
            return False
        short = all(len(t.split()) <= 5 for t in recent)
        punctuated = sum(1 for t in recent if t.strip().endswith(("?", "!"))) >= 2
        return short and punctuated

    def _high_arousal_run(self, user_turns: list[str]) -> bool:
        return len(user_turns) >= 2 and all(self._arousal(t) >= 0.5 for t in user_turns[-2:])

    # --- subtext ----------------------------------------------------------

    def _subtext(self, user_turns: list[str], annotated: dict) -> str:
        current = user_turns[-1] if user_turns else ""
        low = current.lower()

        if self._has(current, CORRECTION_MARKERS):
            return "correction"
        if self._has(current, DISCLOSURE_MARKERS):
            return "disclosure_request"
        if annotated.get("subtext") in {"exhaustion", "resignation"}:
            return annotated["subtext"]
        if self._escalating(user_turns):
            return "escalating"
        if self._demand_pattern(user_turns):
            return "demand"

        stripped = current.rstrip()
        if stripped.endswith("?") and not self._has(current, ESCALATION_MARKERS):
            return "question"
        if stripped.endswith(("!", "?")) or self._has(current, ESCALATION_MARKERS):
            return "challenge"
        return "statement"

    def _stance(self, current: str, annotated: dict) -> str:
        """Whether the user is in the right. Not inferable from keywords —
        this is a semantic judgement, so it defaults to unknown and expects an
        annotation. Guessing here is how a layer starts confidently wrong."""
        return annotated.get("stance", "unknown")

    # --- entry point ------------------------------------------------------

    def infer(
        self,
        messages: list[dict],
        turn_index: int = 0,
        annotated: dict | None = None,
    ) -> AffectState:
        annotated = annotated or {}
        user_turns = [m["content"] for m in messages if m.get("role") == "user"]

        history = self._history(user_turns)
        current = user_turns[-1] if user_turns else ""
        previous = user_turns[-2] if len(user_turns) > 1 else ""
        arousal = self._arousal(current)

        return AffectState(
            valence=round(0.5 - arousal, 3),
            arousal=arousal,
            escalation_delta=round(arousal - self._arousal(previous), 3) if previous else 0.0,
            stance=self._stance(current, annotated),
            subtext=self._subtext(user_turns, annotated),
            turn_index=turn_index,
            history=history,
        )

    def _history(self, user_turns: list[str]) -> tuple[AffectState, ...]:
        """Per-turn series carrying a real delta, so trend checks have
        something to read. A history of zero-delta states cannot show
        escalation no matter how angry the transcript gets."""
        states = []
        for i, text in enumerate(user_turns):
            prev = self._arousal(user_turns[i - 1]) if i else 0.0
            states.append(
                AffectState(
                    valence=0.0,
                    arousal=self._arousal(text),
                    escalation_delta=round(self._arousal(text) - prev, 3),
                    stance="unknown",
                    subtext="",
                    turn_index=i,
                )
            )
        return tuple(states)


def load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]