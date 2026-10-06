from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GoalCandidate:
    """A structurally observable prior turn that may anchor a short reference.

    Candidate does not mean semantically unresolved or correct. It only means
    the turn is eligible for conservative reference resolution.
    """

    turn_index: int
    role: str
    kind: str
    excerpt: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class DialogueReferenceState:
    """Conservative structural state for short deictic/continuation turns."""

    active: bool
    form: str = "none"
    resolved: bool = False
    target_turn_index: int | None = None
    target_role: str = ""
    target_excerpt: str = ""
    anchor_turn_index: int | None = None
    anchor_excerpt: str = ""
    candidate_stack: tuple[GoalCandidate, ...] = ()
    requires_clarification: bool = False
    confidence: float = 0.0
    evidence_level: str = "structural"
    evidence: tuple[str, ...] = ()


_CONTINUE_FORMS = {
    "devam",
    "devam et",
    "continue",
    "go on",
    "keep going",
}

_EXECUTE_REFERENCE_FORMS = {
    "yap",
    "bunu yap",
    "şunu yap",
    "onu yap",
    "do it",
    "do this",
    "do that",
}

_REVISE_REFERENCE_FORMS = {
    "düzelt",
    "bunu düzelt",
    "şunu düzelt",
    "onu düzelt",
    "fix it",
    "fix this",
    "fix that",
}

_ASSISTANT_OFFER_MARKERS = (
    "istersen",
    "yapabilirim",
    "yapayım",
    "yapabiliriz",
    "devam edebilirim",
    "geçebilirim",
    "sıradaki",
    "sonraki",
    "i can ",
    "i could ",
    "would you like me to",
    "shall i",
    "next i can",
    "next step",
)

_USER_GOAL_MARKERS = (
    "yap",
    "ekle",
    "düzelt",
    "kontrol et",
    "bak",
    "gönder",
    "oluştur",
    "sil",
    "çalıştır",
    "araştır",
    "incele",
    "durum ne",
    "son durum",
    "neden",
    "nasıl",
    "anlat",
    "açıkla",
    "do ",
    "fix ",
    "check ",
    "run ",
    "explain ",
    "why ",
    "how ",
)


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _excerpt(text: str) -> str:
    return " ".join(text.strip().split())[:280]


def _previous_index(
    messages: list[dict],
    before: int,
    *,
    role: str | None = None,
) -> int | None:
    for index in range(before - 1, -1, -1):
        if role is None or messages[index].get("role") == role:
            return index
    return None


def _has_offer(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ASSISTANT_OFFER_MARKERS)


def _looks_like_user_goal(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return False
    if text.rstrip().endswith("?"):
        return True
    return any(marker in low for marker in _USER_GOAL_MARKERS)


def _candidate_stack(
    messages: list[dict],
    current_user_index: int,
) -> tuple[GoalCandidate, ...]:
    candidates: list[GoalCandidate] = []

    assistant_index = _previous_index(
        messages,
        current_user_index,
        role="assistant",
    )
    if assistant_index is not None:
        assistant_text = str(messages[assistant_index].get("content", ""))
        if _has_offer(assistant_text):
            candidates.append(
                GoalCandidate(
                    turn_index=assistant_index,
                    role="assistant",
                    kind="assistant_offer",
                    excerpt=_excerpt(assistant_text),
                    evidence=("recent_assistant_offer_marker",),
                )
            )

    for index in range(current_user_index - 1, -1, -1):
        if messages[index].get("role") != "user":
            continue
        text = str(messages[index].get("content", ""))
        if _looks_like_user_goal(text):
            candidates.append(
                GoalCandidate(
                    turn_index=index,
                    role="user",
                    kind="prior_user_goal",
                    excerpt=_excerpt(text),
                    evidence=("recent_user_goal_marker",),
                )
            )
            break

    return tuple(
        sorted(
            candidates,
            key=lambda item: item.turn_index,
            reverse=True,
        )
    )


def _resolve(
    *,
    form: str,
    target_index: int,
    messages: list[dict],
    stack: tuple[GoalCandidate, ...],
    evidence: tuple[str, ...],
) -> DialogueReferenceState:
    target = messages[target_index]
    anchor_index = None
    anchor_excerpt = ""
    if target.get("role") == "assistant":
        anchor_index = _previous_index(messages, target_index, role="user")
        if anchor_index is not None:
            anchor_excerpt = _excerpt(
                str(messages[anchor_index].get("content", ""))
            )

    return DialogueReferenceState(
        active=True,
        form=form,
        resolved=True,
        target_turn_index=target_index,
        target_role=str(target.get("role", "")),
        target_excerpt=_excerpt(str(target.get("content", ""))),
        anchor_turn_index=anchor_index,
        anchor_excerpt=anchor_excerpt,
        candidate_stack=stack,
        requires_clarification=False,
        confidence=1.0,
        evidence=("short_reference_form", *evidence),
    )


def _unresolved(
    *,
    form: str,
    stack: tuple[GoalCandidate, ...],
    evidence: tuple[str, ...],
) -> DialogueReferenceState:
    return DialogueReferenceState(
        active=True,
        form=form,
        resolved=False,
        candidate_stack=stack,
        requires_clarification=True,
        confidence=0.0,
        evidence=("short_reference_form", *evidence),
    )


def infer_dialogue_reference(messages: list[dict]) -> DialogueReferenceState:
    """Resolve only a narrow set of short references with structural evidence.

    This is QUD-inspired bookkeeping, not a full semantic QUD model. It refuses
    to resolve stale/ambiguous `do it` forms and never uses a candidate as proof
    that a goal is truly unresolved.
    """

    if not messages:
        return DialogueReferenceState(
            active=False,
            evidence=("empty_transcript",),
        )

    current_user_index = _previous_index(
        messages,
        len(messages),
        role="user",
    )
    if current_user_index is None:
        return DialogueReferenceState(
            active=False,
            evidence=("no_user_turn",),
        )

    current = _normalise(
        str(messages[current_user_index].get("content", ""))
    )
    stack = _candidate_stack(messages, current_user_index)

    if current in _CONTINUE_FORMS:
        form = "continue"
    elif current in _EXECUTE_REFERENCE_FORMS:
        form = "execute_reference"
    elif current in _REVISE_REFERENCE_FORMS:
        form = "revise_reference"
    else:
        return DialogueReferenceState(
            active=False,
            candidate_stack=stack,
            evidence=("no_supported_short_reference",),
        )

    assistant_index = _previous_index(
        messages,
        current_user_index,
        role="assistant",
    )
    prior_user_index = _previous_index(
        messages,
        current_user_index,
        role="user",
    )

    if assistant_index is None:
        return _unresolved(
            form=form,
            stack=stack,
            evidence=("no_prior_assistant_turn",),
        )

    assistant_text = str(messages[assistant_index].get("content", ""))
    assistant_has_offer = _has_offer(assistant_text)

    if form == "execute_reference":
        if assistant_has_offer:
            return _resolve(
                form=form,
                target_index=assistant_index,
                messages=messages,
                stack=stack,
                evidence=("immediate_assistant_offer",),
            )
        return _unresolved(
            form=form,
            stack=stack,
            evidence=("no_immediate_assistant_offer",),
        )

    if form == "revise_reference":
        return _resolve(
            form=form,
            target_index=assistant_index,
            messages=messages,
            stack=stack,
            evidence=("immediate_assistant_output",),
        )

    if assistant_has_offer:
        return _resolve(
            form=form,
            target_index=assistant_index,
            messages=messages,
            stack=stack,
            evidence=("immediate_assistant_offer",),
        )

    if "?" in assistant_text:
        return _unresolved(
            form=form,
            stack=stack,
            evidence=("assistant_question_blocks_implicit_continue",),
        )

    if prior_user_index is not None:
        prior_user_text = str(
            messages[prior_user_index].get("content", "")
        )
        if _looks_like_user_goal(prior_user_text):
            return _resolve(
                form=form,
                target_index=prior_user_index,
                messages=messages,
                stack=stack,
                evidence=("recent_prior_user_goal",),
            )

    return _unresolved(
        form=form,
        stack=stack,
        evidence=("no_safe_goal_anchor",),
    )
