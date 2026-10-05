from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class IntentState:
    """Conservative, inspectable representation of what the user is asking for."""

    kind: str
    canonical_request: str
    response_mode: str
    confidence: float
    explicit: bool
    needs_clarification: bool
    constraints: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


class IntentAdapter(Protocol):
    name: str

    def infer(self, messages: list[dict]) -> IntentState: ...


ACTION_MARKERS = (
    "yap",
    "ekle",
    "düzelt",
    "kontrol et",
    "bak",
    "gönder",
    "oluştur",
    "sil",
    "devam et",
    "çalıştır",
    "araştır",
    "incele",
)

STATUS_MARKERS = (
    "durum ne",
    "son durum",
    "ne kadar kaldı",
    "bitti mi",
    "oluyor mu",
    "çalışıyor mu",
)

EXPLANATION_MARKERS = (
    "neden",
    "niçin",
    "nasıl",
    "anlat",
    "açıkla",
    "ne demek",
)

VAGUE_FORMS = (
    "bu",
    "bunu",
    "şu",
    "şunu",
    "onu",
    "şey",
)

CONSTRAINT_MARKERS = (
    ("kısa", "brief"),
    ("net", "direct"),
    ("sadece", "scope_limited"),
    ("detaylı", "detailed"),
    ("tahmin etme", "no_guess"),
    ("uydurma", "no_guess"),
)


@dataclass
class HeuristicIntent:
    name: str = "heuristic"

    def infer(self, messages: list[dict]) -> IntentState:
        user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
        current = user_turns[-1].strip() if user_turns else ""
        low = self._normalise(current)
        constraints = self._constraints(low)

        if not current:
            return self._state(
                "unknown", current, "clarify", 0.0,
                explicit=False, needs_clarification=True,
                constraints=constraints, evidence=("empty_user_turn",),
            )

        if self._is_vague(low):
            return self._state(
                "unknown", current, "clarify", 0.2,
                explicit=False, needs_clarification=True,
                constraints=constraints, evidence=("vague_deictic",),
            )

        marker = self._first_marker(low, STATUS_MARKERS)
        if marker:
            return self._state(
                "status_check", current, "direct", 0.94,
                explicit=True, needs_clarification=False,
                constraints=constraints, evidence=(f"status:{marker}",),
            )

        marker = self._first_marker(low, ACTION_MARKERS)
        if marker:
            return self._state(
                "action_request", current, "execute", 0.90,
                explicit=True, needs_clarification=False,
                constraints=constraints, evidence=(f"action:{marker}",),
            )

        marker = self._first_marker(low, EXPLANATION_MARKERS)
        if marker:
            return self._state(
                "explanation", current, "explain", 0.86,
                explicit=True, needs_clarification=False,
                constraints=constraints, evidence=(f"explain:{marker}",),
            )

        if current.rstrip().endswith("?"):
            return self._state(
                "question", current, "direct", 0.74,
                explicit=True, needs_clarification=False,
                constraints=constraints, evidence=("question_mark",),
            )

        return self._state(
            "statement", current, "contextual", 0.50,
            explicit=False, needs_clarification=False,
            constraints=constraints, evidence=("no_explicit_request_marker",),
        )

    def _state(
        self,
        kind: str,
        original: str,
        response_mode: str,
        confidence: float,
        *,
        explicit: bool,
        needs_clarification: bool,
        constraints: tuple[str, ...],
        evidence: tuple[str, ...],
    ) -> IntentState:
        return IntentState(
            kind=kind,
            canonical_request=self._canonical(kind, original, constraints),
            response_mode=response_mode,
            confidence=confidence,
            explicit=explicit,
            needs_clarification=needs_clarification,
            constraints=constraints,
            evidence=evidence,
        )

    def _canonical(self, kind: str, original: str, constraints: tuple[str, ...]) -> str:
        prefix = {
            "action_request": "Perform the explicit user request",
            "status_check": "Check the current status and report the result",
            "explanation": "Explain the requested point",
            "question": "Answer the user's question directly",
            "statement": "Respond to the user's statement without inventing a request",
            "unknown": "Clarify what the user wants before acting",
        }[kind]
        request = original.strip() or "<empty>"
        suffix = f" Constraints: {', '.join(constraints)}." if constraints else ""
        return f'{prefix}: "{request}".{suffix}'

    def _normalise(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _first_marker(self, text: str, markers: tuple[str, ...]) -> str | None:
        for marker in markers:
            if re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", text):
                return marker
        return None

    def _is_vague(self, text: str) -> bool:
        bare = text.strip(" ?!.,;:")
        return bare in VAGUE_FORMS and len(bare.split()) <= 2

    def _constraints(self, text: str) -> tuple[str, ...]:
        found = []
        for marker, constraint in CONSTRAINT_MARKERS:
            if marker in text and constraint not in found:
                found.append(constraint)
        return tuple(found)
