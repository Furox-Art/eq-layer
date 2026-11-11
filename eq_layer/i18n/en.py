"""English (EN) dialogue markers, lexicons, and patterns for EQ-Layer."""

from __future__ import annotations

# English was not mapped into heuristic intent markers in original eq-layer
ACTION_MARKERS: tuple[str, ...] = ()
STATUS_MARKERS: tuple[str, ...] = ()
EXPLANATION_MARKERS: tuple[str, ...] = ()
VAGUE_FORMS: tuple[str, ...] = ()
CONSTRAINT_MARKERS: tuple[tuple[str, str], ...] = ()

ESCALATION_MARKERS: tuple[str, ...] = ()
SOFTENERS: tuple[str, ...] = ()
QUESTION_OPENERS: tuple[str, ...] = ()

CORRECTION_MARKERS: tuple[str, ...] = (
    "that's not what i meant",
    "you misunderstood",
    "no, i meant",
    "i didn't say that",
    "i said",
    "i meant",
    "meant to say",
    "that doesn't work",
    "that does not work",
    "that doesn’t work",
    "that's wrong",
    "that is wrong",
)

DISCLOSURE_MARKERS: tuple[str, ...] = ()

CONTINUE_FORMS: set[str] = {
    "continue",
    "go on",
    "keep going",
}

EXECUTE_REFERENCE_FORMS: set[str] = {
    "do it",
    "do this",
    "do that",
}

REVISE_REFERENCE_FORMS: set[str] = {
    "fix it",
    "fix this",
    "fix that",
}

ASSISTANT_OFFER_MARKERS: tuple[str, ...] = (
    "would you like me to",
    "shall i",
)

USER_GOAL_MARKERS: tuple[str, ...] = (
    "do",
    "fix",
    "check",
    "run",
    "explain",
    "why",
    "how",
)

ABSOLUTE_PRIVACY_PROMISES: tuple[str, ...] = (
    "i guarantee confidentiality",
    "this will stay between us",
    "no one else will know",
    "completely confidential",
)

STANCE_ADMISSIONS: tuple[str, ...] = (
    "you're right",
    "you are right",
    "i was wrong",
    "my mistake",
)

GENERIC_PHRASES: tuple[str, ...] = (
    "i understand how you feel",
    "that sounds really tough",
    "that must be frustrating",
    "i'm sorry for your frustration",
    "that sounds incredibly difficult",
    "your feelings are valid",
    "i hear you",
    "that's completely understandable",
)
