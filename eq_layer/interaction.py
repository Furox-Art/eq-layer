from __future__ import annotations

import re
from dataclasses import dataclass

from .affect import CORRECTION_MARKERS


@dataclass(frozen=True)
class RepairState:
    """Structural conversational repair state.

    This state does not guess factual correctness. It only records observable
    evidence that the user is correcting the assistant or that repair is
    repeating.
    """

    active: bool
    kind: str = "none"
    target_turn_index: int | None = None
    target_excerpt: str = ""
    correction_excerpt: str = ""
    replacement_excerpt: str = ""
    replacement_explicit: bool = False
    repeated: bool = False
    recent_repair_count: int = 0
    confidence: float = 0.0
    evidence_level: str = "structural"
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class InteractionQualityState:
    """Conversation-health signal kept separate from user affect."""

    current: float
    delta: float
    repeated_failure_count: int
    unresolved_repair_count: int
    clarification_count: int
    evidence_level: str = "structural"
    evidence: tuple[str, ...] = ()


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _has_correction(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in CORRECTION_MARKERS)


def _clean_replacement(value: str) -> str:
    candidate = value.strip(" \t\r\n.,;:!?-–—")
    normalized = _normalise(candidate)
    trivial = {
        "",
        "that",
        "it",
        "this",
        "that one",
        "bu",
        "şu",
        "o",
        "onu",
        "bunu",
        "şunu",
    }
    if normalized in trivial:
        return ""
    return candidate[:240]


def _extract_replacement(text: str) -> str:
    """Extract an explicitly supplied replacement without inferring one.

    The final matching correction phrase wins, so
    "I did not say X; I said Y" yields Y. This is structural extraction only:
    the replacement is not treated as factually correct.
    """

    patterns = (
        r"\bi\s+said\b\s*(?:[:,\-–—]\s*)?(.+)$",
        r"\bi\s+meant(?:\s+to\s+say)?\b\s*(?:[:,\-–—]\s*)?(.+)$",
        r"\bmeant\s+to\s+say\b\s*(?:[:,\-–—]\s*)?(.+)$",
        r"\bdemek\s+istediğim\b\s*(?:[:,\-–—]\s*)?(.+)$",
        r"\bkastettiğim\b\s*(?:[:,\-–—]\s*)?(.+)$",
        r"(?:\bhayır\s*[,;:]?\s*)?(?:\bben\s+)?\bonu\s+demedim\b[.!?;,:\-–—\s]*(.+)$",
        r"(?:\bhayır\s*[,;:]?\s*)?(?:\bben\s+)?\bonu\s+sormadım\b[.!?;,:\-–—\s]*(.+)$",
    )

    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            replacement = _clean_replacement(match.group(1))
            if replacement:
                matches.append((match.start(), replacement))

    if not matches:
        return ""
    return max(matches, key=lambda item: item[0])[1]


def _last_assistant_before(messages: list[dict], index: int) -> int | None:
    for cursor in range(index - 1, -1, -1):
        if messages[cursor].get("role") == "assistant":
            return cursor
    return None


def _correction_events(messages: list[dict]) -> list[tuple[int, int]]:
    events: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        text = str(message.get("content", ""))
        if not _has_correction(text):
            continue
        assistant_index = _last_assistant_before(messages, index)
        if assistant_index is not None:
            events.append((index, assistant_index))
    return events


def _clarification_count(messages: list[dict]) -> int:
    """Count assistant turns that visibly ask for clarification.

    A question mark is only treated as a structural proxy. It is not labelled
    as an error by itself; it contributes to interaction cost only when the
    conversation already contains repair pressure.
    """

    count = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        text = str(message.get("content", ""))
        if "?" not in text:
            continue
        if any(
            previous.get("role") == "user"
            for previous in messages[:index]
        ):
            count += 1
    return count


def infer_repair_state(messages: list[dict]) -> RepairState:
    events = _correction_events(messages)
    if not messages:
        return RepairState(active=False, evidence=("empty_transcript",))

    last_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        None,
    )
    if last_user_index is None:
        return RepairState(active=False, evidence=("no_user_turn",))

    current_event = next(
        (
            (user_index, assistant_index)
            for user_index, assistant_index in reversed(events)
            if user_index == last_user_index
        ),
        None,
    )
    recent_events = [
        event
        for event in events
        if event[0] >= max(0, last_user_index - 6)
    ]

    if current_event is None:
        return RepairState(
            active=False,
            recent_repair_count=len(recent_events),
            evidence=("no_current_user_correction",),
        )

    user_index, assistant_index = current_event

    # A correction is unresolved only until the assistant has responded after
    # that correction. Historical corrections remain evidence of interaction
    # quality but are not kept artificially "active".
    user_text = str(messages[user_index].get("content", "")).strip()
    replacement = _extract_replacement(user_text)

    if any(
        message.get("role") == "assistant"
        for message in messages[user_index + 1:]
    ):
        return RepairState(
            active=False,
            kind="responded_repair",
            target_turn_index=assistant_index,
            correction_excerpt=user_text[:240],
            replacement_excerpt=replacement,
            replacement_explicit=bool(replacement),
            repeated=len(recent_events) >= 2,
            recent_repair_count=len(recent_events),
            confidence=1.0,
            evidence_level="structural",
            evidence=("assistant_response_after_correction",),
        )

    repeated = len(recent_events) >= 2

    assistant_text = str(messages[assistant_index].get("content", "")).strip()

    return RepairState(
        active=True,
        kind="user_corrects_assistant",
        target_turn_index=assistant_index,
        target_excerpt=assistant_text[:240],
        correction_excerpt=user_text[:240],
        replacement_excerpt=replacement,
        replacement_explicit=bool(replacement),
        repeated=repeated,
        recent_repair_count=len(recent_events),
        confidence=1.0,
        evidence_level="structural",
        evidence=(
            "explicit_user_correction_marker",
            "prior_assistant_turn",
            *(("explicit_replacement",) if replacement else ()),
            *(
                ("repeated_recent_repair",)
                if repeated
                else ()
            ),
        ),
    )


def _quality_components(messages: list[dict]) -> tuple[float, int, int, int, tuple[str, ...]]:
    events = _correction_events(messages)
    repair = infer_repair_state(messages)
    clarification_count = _clarification_count(messages)

    # Repeated failure means multiple structurally observed user->assistant
    # repairs close together. It does not assert that the user is factually right.
    repeated_failure_count = max(0, repair.recent_repair_count - 1)

    # A current user correction has not yet received an assistant repair turn.
    unresolved_repair_count = 1 if repair.active else 0

    penalty = 0.0
    evidence: list[str] = []

    if unresolved_repair_count:
        penalty += 0.30
        evidence.append("active_unresolved_repair")

    if repeated_failure_count:
        penalty += min(0.30, 0.15 * repeated_failure_count)
        evidence.append("repeated_recent_repair")

    # Clarifications are not intrinsically bad. Penalise only excess
    # clarification in a transcript that is already under repair pressure.
    if events and clarification_count > 1:
        penalty += min(0.15, 0.05 * (clarification_count - 1))
        evidence.append("clarification_overhead_during_repair")

    score = round(max(0.0, min(1.0, 1.0 - penalty)), 3)
    if not evidence:
        evidence.append("no_structural_failure_signal")

    return (
        score,
        repeated_failure_count,
        unresolved_repair_count,
        clarification_count,
        tuple(evidence),
    )


def infer_interaction_quality(messages: list[dict]) -> InteractionQualityState:
    (
        current,
        repeated_failure_count,
        unresolved_repair_count,
        clarification_count,
        evidence,
    ) = _quality_components(messages)

    last_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        None,
    )

    previous = 1.0
    if last_user_index is not None and last_user_index > 0:
        previous_messages = messages[:last_user_index]
        previous = _quality_components(previous_messages)[0]

    return InteractionQualityState(
        current=current,
        delta=round(current - previous, 3),
        repeated_failure_count=repeated_failure_count,
        unresolved_repair_count=unresolved_repair_count,
        clarification_count=clarification_count,
        evidence_level="structural",
        evidence=evidence,
    )
