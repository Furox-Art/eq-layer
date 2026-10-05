from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher


ABSOLUTE_PRIVACY_PROMISES = (
    "kimseye söylemem",
    "sadece bu sohbette kalır",
    "aramızda kalacak",
    "tamamen gizli kalacak",
    "i guarantee confidentiality",
    "this will stay between us",
    "no one else will know",
    "completely confidential",
)

STANCE_ADMISSIONS = (
    "haklısın",
    "sen haklısın",
    "ben yanıldım",
    "ben yanlış anladım",
    "ben hatalıydım",
    "you're right",
    "you are right",
    "i was wrong",
    "my mistake",
)


@dataclass(frozen=True)
class AuditIssue:
    code: str
    severity: str
    message: str
    evidence: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AuditContext:
    task_move: str
    social_move: str
    repair_move: str
    verbosity: str
    question_budget: int
    no_guess: bool
    stance: str = "unknown"
    repair_active: bool = False
    previous_assistant_responses: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResponseAudit:
    passed: bool
    should_regenerate: bool
    hard_failures: int
    soft_warnings: int
    question_count: int
    word_count: int
    max_repeat_similarity: float
    issues: tuple[AuditIssue, ...]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "should_regenerate": self.should_regenerate,
            "hard_failures": self.hard_failures,
            "soft_warnings": self.soft_warnings,
            "question_count": self.question_count,
            "word_count": self.word_count,
            "max_repeat_similarity": self.max_repeat_similarity,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _sentence_starts(text: str) -> tuple[str, ...]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", _normalise(text))
    return tuple(part.strip(" -—:;,.!?") for part in parts if part.strip())


def _max_repeat_similarity(
    response: str,
    previous_assistant_responses: tuple[str, ...],
) -> float:
    current = _normalise(response)
    if not current or not previous_assistant_responses:
        return 0.0

    scores = []
    for previous in previous_assistant_responses[-3:]:
        previous_norm = _normalise(previous)
        if not previous_norm:
            continue
        scores.append(SequenceMatcher(None, current, previous_norm).ratio())
    return round(max(scores, default=0.0), 4)


class ResponseAuditor:
    """Conservative structural auditor for generated responses.

    This auditor checks inspectable control-contract violations. It does not
    decide whether factual content is true, whether advice is correct, or
    whether the user's stance is justified.
    """

    def audit(self, response: str, context: AuditContext) -> ResponseAudit:
        issues: list[AuditIssue] = []
        stripped = response.strip()
        low = _normalise(stripped)
        question_count = stripped.count("?")
        word_count = len(stripped.split())
        repeat_similarity = _max_repeat_similarity(
            stripped,
            context.previous_assistant_responses,
        )

        if not stripped:
            issues.append(
                AuditIssue(
                    code="empty_response",
                    severity="hard",
                    message="The model returned an empty response.",
                )
            )

        if question_count > context.question_budget:
            issues.append(
                AuditIssue(
                    code="question_budget_exceeded",
                    severity="hard",
                    message=(
                        f"Response asked {question_count} questions but the "
                        f"control action allowed {context.question_budget}."
                    ),
                    evidence=f"question_count={question_count}",
                )
            )

        clarification_required = (
            context.task_move == "clarify_goal"
            or context.repair_move in {"clarify_goal", "clarify_one"}
        )
        if clarification_required and question_count == 0:
            issues.append(
                AuditIssue(
                    code="missing_required_clarification",
                    severity="hard",
                    message="The control action required a targeted clarification question.",
                )
            )

        for phrase in ABSOLUTE_PRIVACY_PROMISES:
            if phrase in low:
                issues.append(
                    AuditIssue(
                        code="absolute_privacy_promise",
                        severity="hard",
                        message="Response made an unsupported absolute privacy promise.",
                        evidence=phrase,
                    )
                )
                break

        if context.stance == "unknown":
            starts = _sentence_starts(stripped)
            stance_hit = next(
                (
                    phrase
                    for phrase in STANCE_ADMISSIONS
                    if any(
                        sentence == phrase
                        or sentence.startswith(phrase + " ")
                        or sentence.startswith(phrase + ",")
                        for sentence in starts
                    )
                ),
                None,
            )
            if stance_hit is not None:
                issues.append(
                    AuditIssue(
                        code="unsupported_stance_admission",
                        severity="hard",
                        message=(
                            "Response admitted that the user was right or the "
                            "assistant was wrong while factual stance was unknown."
                        ),
                        evidence=stance_hit,
                    )
                )

        if (
            context.verbosity == "low"
            and word_count > 120
        ):
            issues.append(
                AuditIssue(
                    code="low_verbosity_overrun",
                    severity="soft",
                    message="Response substantially exceeded the low-verbosity target.",
                    evidence=f"word_count={word_count}",
                )
            )

        if repeat_similarity >= 0.92 and word_count >= 8:
            issues.append(
                AuditIssue(
                    code="near_duplicate_response",
                    severity="soft",
                    message="Response is nearly identical to a recent assistant response.",
                    evidence=f"similarity={repeat_similarity:.4f}",
                )
            )

        if (
            context.repair_active
            and context.repair_move in {"repair_targeted", "stop_restatement_and_repair"}
            and repeat_similarity >= 0.80
        ):
            issues.append(
                AuditIssue(
                    code="repair_repeats_prior_answer",
                    severity="hard",
                    message=(
                        "Repair was active but the response substantially repeated "
                        "a recent assistant answer."
                    ),
                    evidence=f"similarity={repeat_similarity:.4f}",
                )
            )

        hard = sum(issue.severity == "hard" for issue in issues)
        soft = sum(issue.severity == "soft" for issue in issues)
        return ResponseAudit(
            passed=hard == 0,
            should_regenerate=hard > 0,
            hard_failures=hard,
            soft_warnings=soft,
            question_count=question_count,
            word_count=word_count,
            max_repeat_similarity=repeat_similarity,
            issues=tuple(issues),
        )


def audit_from_generation_metadata(
    response: str,
    messages: list[dict],
    metadata: dict,
) -> ResponseAudit:
    """Audit an EQ-Layer response using generation metadata only."""

    factored = metadata.get("factored_action") or {}
    realization = factored.get("realization") or {}
    repair = metadata.get("repair") or {}
    previous = tuple(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "assistant"
    )

    context = AuditContext(
        task_move=str(factored.get("task_move") or "respond_contextually"),
        social_move=str(factored.get("social_move") or "neutral"),
        repair_move=str(factored.get("repair_move") or "none"),
        verbosity=str(realization.get("verbosity") or "normal"),
        question_budget=int(realization.get("question_budget") or 0),
        no_guess=bool(realization.get("no_guess", False)),
        stance=str(metadata.get("stance") or "unknown"),
        repair_active=bool(repair.get("active", False)),
        previous_assistant_responses=previous,
    )
    return ResponseAuditor().audit(response, context)
