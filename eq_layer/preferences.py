from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace


SESSION_SCOPE_PATTERNS = (
    r"\bbundan sonra\b",
    r"\bbu sohbet(?:te| boyunca)\b",
    r"\bher zaman\b",
    r"\bdaima\b",
    r"\bfrom now on\b",
    r"\bfor this conversation\b",
    r"\bthroughout this conversation\b",
    r"\balways\b",
)

STRONG_TURKISH_STYLE_PATTERN = re.compile(
    r"\bbenimle\b.*\bkonuş\b",
    re.IGNORECASE,
)
STRONG_ENGLISH_STYLE_PATTERN = re.compile(
    r"\b(?:keep|make)\s+(?:your\s+)?responses?\b|\brespond\b.*\b(?:brief|concise|direct)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SessionPreferences:
    """Explicit session-scoped preferences, separate from affect.

    The state is recomputed from the visible transcript. It does not infer
    personality, emotional traits, or cross-session user properties.
    """

    verbosity: str | None = None
    directness: str | None = None
    no_guess: bool = False
    scope_limited: bool = False
    source_turns: tuple[int, ...] = ()
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _session_scoped(text: str) -> bool:
    low = _normalise(text)
    if any(re.search(pattern, low) for pattern in SESSION_SCOPE_PATTERNS):
        return True
    return bool(
        STRONG_TURKISH_STYLE_PATTERN.search(low)
        or STRONG_ENGLISH_STYLE_PATTERN.search(low)
    )


def _updates(text: str) -> dict:
    low = _normalise(text)
    updates: dict[str, object] = {}

    if re.search(r"\b(kısa|öz|brief|concise)\b", low):
        updates["verbosity"] = "low"
    if re.search(r"\b(detaylı|ayrıntılı|detailed)\b|\bin detail\b", low):
        updates["verbosity"] = "high"

    if re.search(r"\b(net|direkt|doğrudan|direct)\b|\bstraight to the point\b", low):
        updates["directness"] = "high"

    if (
        "tahmin etme" in low
        or "uydurma" in low
        or "don't guess" in low
        or "do not guess" in low
    ):
        updates["no_guess"] = True

    if (
        "sadece sorduğuma cevap ver" in low
        or "sadece istediğimi yap" in low
        or "only answer what i ask" in low
        or "only do what i ask" in low
        or "stay within scope" in low
    ):
        updates["scope_limited"] = True

    return updates


def infer_session_preferences(messages: list[dict]) -> SessionPreferences:
    """Infer only explicit, session-scoped style/control preferences.

    Later explicit session preferences override earlier conflicting values.
    Ordinary one-turn constraints are intentionally not persisted.
    """

    state = SessionPreferences()
    source_turns: list[int] = []
    evidence: list[str] = []

    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue

        text = str(message.get("content", "")).strip()
        if not text or not _session_scoped(text):
            continue

        updates = _updates(text)
        if not updates:
            continue

        state = replace(state, **updates)
        source_turns.append(index)
        for key, value in updates.items():
            evidence.append(f"turn:{index}:{key}={value}")

    return replace(
        state,
        source_turns=tuple(source_turns),
        evidence=tuple(evidence),
    )
