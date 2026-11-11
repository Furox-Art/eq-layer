"""Internationalization (i18n) and multilingual lexicon registry for EQ-Layer."""

from __future__ import annotations

from . import en, ru, tr

SUPPORTED_LANGUAGES = ("tr", "en", "ru")

# Merged cross-language marker tuples for backward-compatible global matching
ACTION_MARKERS: tuple[str, ...] = tr.ACTION_MARKERS + en.ACTION_MARKERS + ru.ACTION_MARKERS
STATUS_MARKERS: tuple[str, ...] = tr.STATUS_MARKERS + en.STATUS_MARKERS + ru.STATUS_MARKERS
EXPLANATION_MARKERS: tuple[str, ...] = tr.EXPLANATION_MARKERS + en.EXPLANATION_MARKERS + ru.EXPLANATION_MARKERS
VAGUE_FORMS: tuple[str, ...] = tr.VAGUE_FORMS + en.VAGUE_FORMS + ru.VAGUE_FORMS
CONSTRAINT_MARKERS: tuple[tuple[str, str], ...] = tr.CONSTRAINT_MARKERS + en.CONSTRAINT_MARKERS + ru.CONSTRAINT_MARKERS

ESCALATION_MARKERS: tuple[str, ...] = tr.ESCALATION_MARKERS + en.ESCALATION_MARKERS + ru.ESCALATION_MARKERS
SOFTENERS: tuple[str, ...] = tr.SOFTENERS + en.SOFTENERS + ru.SOFTENERS
CORRECTION_MARKERS: tuple[str, ...] = tr.CORRECTION_MARKERS + en.CORRECTION_MARKERS + ru.CORRECTION_MARKERS
DISCLOSURE_MARKERS: tuple[str, ...] = tr.DISCLOSURE_MARKERS + en.DISCLOSURE_MARKERS + ru.DISCLOSURE_MARKERS
QUESTION_OPENERS: tuple[str, ...] = tr.QUESTION_OPENERS + en.QUESTION_OPENERS + ru.QUESTION_OPENERS

CONTINUE_FORMS: set[str] = tr.CONTINUE_FORMS | en.CONTINUE_FORMS | ru.CONTINUE_FORMS
EXECUTE_REFERENCE_FORMS: set[str] = tr.EXECUTE_REFERENCE_FORMS | en.EXECUTE_REFERENCE_FORMS | ru.EXECUTE_REFERENCE_FORMS
REVISE_REFERENCE_FORMS: set[str] = tr.REVISE_REFERENCE_FORMS | en.REVISE_REFERENCE_FORMS | ru.REVISE_REFERENCE_FORMS
ASSISTANT_OFFER_MARKERS: tuple[str, ...] = tr.ASSISTANT_OFFER_MARKERS + en.ASSISTANT_OFFER_MARKERS + ru.ASSISTANT_OFFER_MARKERS
USER_GOAL_MARKERS: tuple[str, ...] = tr.USER_GOAL_MARKERS + en.USER_GOAL_MARKERS + ru.USER_GOAL_MARKERS

ABSOLUTE_PRIVACY_PROMISES: tuple[str, ...] = tr.ABSOLUTE_PRIVACY_PROMISES + en.ABSOLUTE_PRIVACY_PROMISES + ru.ABSOLUTE_PRIVACY_PROMISES
STANCE_ADMISSIONS: tuple[str, ...] = tr.STANCE_ADMISSIONS + en.STANCE_ADMISSIONS + ru.STANCE_ADMISSIONS
GENERIC_PHRASES: tuple[str, ...] = tr.GENERIC_PHRASES + en.GENERIC_PHRASES + ru.GENERIC_PHRASES

_MODULES = {
    "tr": tr,
    "en": en,
    "ru": ru,
}


def get_lexicon(lang: str = "all"):
    """Retrieve language-specific or merged lexicon module."""
    if lang == "all":
        import eq_layer.i18n as merged
        return merged
    if lang in _MODULES:
        return _MODULES[lang]
    raise ValueError(f"Unsupported language {lang!r}. Supported: {SUPPORTED_LANGUAGES}")
