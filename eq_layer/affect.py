from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Protocol

from .policies import AffectState


class AffectAdapter(Protocol):
    name: str

    def infer(self, messages: list[dict], turn_index: int = 0) -> AffectState: ...


@dataclass
class HeuristicAffect:
    name: str = "heuristic"

    ESCALATION_MARKERS = (
        "neden",
        "sen de",
        "saçmalama",
        "yine",
        "kaç kez",
        "anlamıyorsun",
        "sinir",
        "rezalet",
    )
    SOFTENERS = ("galiba", "aslında", "belki", "biraz", "emin değilim", "kısmen")

    def infer(self, messages: list[dict], turn_index: int = 0) -> AffectState:
        user_turns = [m["content"] for m in messages if m.get("role") == "user"]
        current = user_turns[-1] if user_turns else ""
        previous = user_turns[-2] if len(user_turns) > 1 else ""

        arousal = self._arousal(current)
        delta = arousal - self._arousal(previous) if previous else 0.0

        stance = "user_right" if self._concedes(current) else "user_asserts"
        subtext = self._subtext(current)

        history = tuple(
            AffectState(0.0, a, 0.0, "unknown", "", i) for i, a in enumerate(self._arousal_series(user_turns))
        )
        return AffectState(
            valence=-abs(arousal - 0.5) * 2 + 0.5,
            arousal=arousal,
            escalation_delta=delta,
            stance=stance,
            subtext=subtext,
            turn_index=turn_index,
            history=history,
        )

    def _arousal(self, text: str) -> float:
        if not text:
            return 0.0
        low = text.lower()
        hits = sum(1 for m in self.ESCALATION_MARKERS if m in low)
        softens = sum(1 for s in self.SOFTENERS if s in low)
        score = min(1.0, 0.25 + hits * 0.22 - softens * 0.12)
        if "?" in text or "!" in text:
            score = min(1.0, score + 0.1)
        return round(score, 3)

    def _concedes(self, text: str) -> bool:
        return "haklısın" in text.lower() or "haklı olduğunu" in text.lower()

    def _subtext(self, text: str) -> str:
        return "challenge" if "?" in text else "statement"

    def _arousal_series(self, user_turns: Iterable[str]) -> list[float]:
        return [self._arousal(t) for t in user_turns]


def load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]