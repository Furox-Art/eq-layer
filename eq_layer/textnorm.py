"""Shared text-normalisation helpers.

Five modules used to carry copy-pasted ``_normalise`` helpers, and the copies
had already drifted into two behaviours:

* a light form — lowercase and collapse whitespace, punctuation preserved
  (``auditor``, ``preferences``, ``intent``);
* a plain form — also replace punctuation runs with spaces (``interaction``),
  with ``dialogue_reference`` additionally dropping the Turkish combining dot
  above (U+0307) so dotted and dotless spellings compare equal.

Both behaviours live here, in one place. Callers must keep using the variant
their matching logic was written against: this consolidation removes the
duplicated code without silently changing any matcher's semantics.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCTUATION = re.compile(r"[^\w\s]")
_COMBINING_DOT_ABOVE = "\u0307"
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u180e"), " ")


def normalise_light(text: str) -> str:
    """Lowercase and collapse whitespace. Punctuation is preserved."""
    return " ".join(text.lower().split())


def normalise_plain(text: str, *, drop_combining_dot_above: bool = False) -> str:
    """Lowercase, replace punctuation with spaces, and collapse whitespace.

    ``drop_combining_dot_above`` removes U+0307 before the punctuation pass so
    that ``i`` + combining dot above normalises like a bare ``i``. This is
    needed by Turkish dotted/dotless matching in the dialogue-reference
    matcher; it is off by default so the default behaviour stays identical to
    the historical copies.
    """
    lowered = text.lower()
    if drop_combining_dot_above:
        lowered = lowered.replace(_COMBINING_DOT_ABOVE, "")
    return " ".join(_PUNCTUATION.sub(" ", lowered).split())


def normalise_match(text: str) -> str:
    """Fold text for phrase matching, erasing spelling-level evasions.

    Unlike :func:`normalise_plain`, this also applies NFKD compatibility
    folding, casefolding, combining-mark removal and Turkish dotless-i
    folding (``ı`` to ``i``). It is deliberately aggressive: it is only used
    to decide whether a response contains a known contract-violating phrase,
    so ``KİMSEYE-SÖYLEMEM``, ``ｋｉｍｓｅｙｅ söylemem``, ``sen haklisin`` and
    ``you’re right`` must all match their canonical spellings. Punctuation
    becomes spaces so the caller can enforce word boundaries on the result.
    """
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.replace("ı", "i").translate(_ZERO_WIDTH)
    return " ".join(_PUNCTUATION.sub(" ", folded).split())
