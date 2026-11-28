"""Unit tests for the modular i18n lexicon package."""

from __future__ import annotations

import pytest

from eq_layer import i18n
from eq_layer.i18n import get_lexicon, SUPPORTED_LANGUAGES


def test_supported_languages() -> None:
    assert "ru" in SUPPORTED_LANGUAGES
    assert "en" in SUPPORTED_LANGUAGES
    assert "tr" in SUPPORTED_LANGUAGES


@pytest.mark.parametrize("lang", ["ru", "en", "tr"])
def test_language_lexicons_have_required_attributes(lang: str) -> None:
    lex = get_lexicon(lang)
    assert hasattr(lex, "ACTION_MARKERS")
    assert hasattr(lex, "STATUS_MARKERS")
    assert hasattr(lex, "EXPLANATION_MARKERS")
    assert hasattr(lex, "VAGUE_FORMS")
    assert hasattr(lex, "CONSTRAINT_MARKERS")
    assert hasattr(lex, "ESCALATION_MARKERS")
    assert hasattr(lex, "SOFTENERS")
    assert hasattr(lex, "CORRECTION_MARKERS")
    assert hasattr(lex, "DISCLOSURE_MARKERS")
    assert hasattr(lex, "QUESTION_OPENERS")
    assert hasattr(lex, "CONTINUE_FORMS")
    assert hasattr(lex, "EXECUTE_REFERENCE_FORMS")
    assert hasattr(lex, "REVISE_REFERENCE_FORMS")
    assert hasattr(lex, "ASSISTANT_OFFER_MARKERS")
    assert hasattr(lex, "USER_GOAL_MARKERS")
    assert hasattr(lex, "ABSOLUTE_PRIVACY_PROMISES")
    assert hasattr(lex, "STANCE_ADMISSIONS")
    assert hasattr(lex, "GENERIC_PHRASES")


def test_get_lexicon_all() -> None:
    merged = get_lexicon("all")
    assert "yap" in merged.ACTION_MARKERS
    assert "сделай" in merged.ACTION_MARKERS
    assert "do" in merged.USER_GOAL_MARKERS


def test_get_lexicon_invalid() -> None:
    with pytest.raises(ValueError, match="Unsupported language"):
        get_lexicon("invalid_lang")
