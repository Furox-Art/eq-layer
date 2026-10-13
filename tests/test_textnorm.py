"""Regression tests for the shared text normalisation (audit H5).

`_normalise` was copy-pasted across five modules and had drifted into two
behaviours. The helpers now live in ``eq_layer.textnorm``; these tests freeze
the exact per-variant behaviour on canonical cases (including the Turkish
dotted/dotless-I rule) and assert the modules really share one implementation.
"""

from __future__ import annotations

import inspect

import pytest

from eq_layer import (
    auditor,
    dialogue_reference,
    interaction,
    intent,
    preferences,
    textnorm,
)

# Expected values below were captured from the pre-refactor implementations
# (lowercase + whitespace collapse for the light variant; punctuation-to-space
# for the plain variant; plain + U+0307 removal for dialogue references).
LIGHT_EXPECTED = {
    "  Hello   World  ": "hello world",
    "Mixed CASE Text": "mixed case text",
    "punctuation, here! really?": "punctuation, here! really?",
    "tabs\tand\nnewlines": "tabs and newlines",
    "\u0130stanbul": "i\u0307stanbul",
    "i\u0307": "i\u0307",
    "": "",
    "   ": "",
    "emoji \U0001F600 text": "emoji \U0001F600 text",
    "under_score and-hyphen": "under_score and-hyphen",
    "What about THIS_one?": "what about this_one?",
    "na\u00efve caf\u00e9": "na\u00efve caf\u00e9",
    "\u0130\u0307 mixed": "i\u0307\u0307 mixed",
}

PLAIN_EXPECTED = {
    "  Hello   World  ": "hello world",
    "Mixed CASE Text": "mixed case text",
    "punctuation, here! really?": "punctuation here really",
    "tabs\tand\nnewlines": "tabs and newlines",
    "\u0130stanbul": "i stanbul",
    "i\u0307": "i",
    "": "",
    "   ": "",
    "emoji \U0001F600 text": "emoji text",
    "under_score and-hyphen": "under_score and hyphen",
    "What about THIS_one?": "what about this_one",
    "na\u00efve caf\u00e9": "na\u00efve caf\u00e9",
    "\u0130\u0307 mixed": "i mixed",
}

# The dialogue-reference matcher additionally drops the combining dot above.
TURKISH_EXPECTED = {
    **PLAIN_EXPECTED,
    "\u0130stanbul": "istanbul",
    "i\u0307": "i",
    "\u0130\u0307 mixed": "i mixed",
}


@pytest.mark.parametrize("case", sorted(LIGHT_EXPECTED))
def test_light_variant_parity(case: str) -> None:
    expected = LIGHT_EXPECTED[case]
    assert textnorm.normalise_light(case) == expected
    assert auditor._normalise(case) == expected
    assert preferences._normalise(case) == expected
    assert intent.HeuristicIntent()._normalise(case) == expected


@pytest.mark.parametrize("case", sorted(PLAIN_EXPECTED))
def test_plain_variant_parity(case: str) -> None:
    expected = PLAIN_EXPECTED[case]
    assert textnorm.normalise_plain(case) == expected
    assert interaction._normalise(case) == expected


@pytest.mark.parametrize("case", sorted(TURKISH_EXPECTED))
def test_turkish_dotted_i_variant_parity(case: str) -> None:
    expected = TURKISH_EXPECTED[case]
    assert textnorm.normalise_plain(case, drop_combining_dot_above=True) == expected
    assert dialogue_reference._normalise(case) == expected


def test_modules_import_the_same_symbols() -> None:
    # Light and plain users must bind the shared functions themselves.
    assert auditor._normalise is textnorm.normalise_light
    assert preferences._normalise is textnorm.normalise_light
    assert interaction._normalise is textnorm.normalise_plain
    assert dialogue_reference.normalise_plain is textnorm.normalise_plain


def test_no_copy_pasted_normalise_left() -> None:
    for module in (auditor, preferences, interaction):
        assert "def _normalise" not in inspect.getsource(module)
    # dialogue_reference keeps exactly one documented thin wrapper that
    # delegates to the shared implementation.
    wrapper = inspect.getsource(dialogue_reference._normalise)
    assert "normalise_plain" in wrapper
    assert "drop_combining_dot_above=True" in wrapper
