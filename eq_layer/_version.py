"""Single source of truth for the EQ-Layer Python package version.

The editable location is the ``version`` field of the ``[project]`` table in
``pyproject.toml``. When the distribution is installed, ``importlib.metadata``
is authoritative; from a source checkout the same field is read from the
nearest ``pyproject.toml`` so both paths report the identical string instead
of a hardcoded copy that drifts (audit M4).
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DISTRIBUTION = "eq-layer"

_PROJECT_HEADER = re.compile(r"(?m)^\[project\]\s*$")
_NEXT_TABLE = re.compile(r"(?m)^\[")
_VERSION_KEY = re.compile(r'(?m)^version\s*=\s*"([^"]+)"')


def _project_table(pyproject_text: str) -> str:
    """Return the ``[project]`` table body, or "" when absent.

    Scoping the search matters: an unrelated ``version =`` key under another
    table must not win over the package version.
    """
    header = _PROJECT_HEADER.search(pyproject_text)
    if header is None:
        return ""
    rest = pyproject_text[header.end() :]
    next_table = _NEXT_TABLE.search(rest)
    return rest[: next_table.start()] if next_table else rest


def _source_checkout_version() -> str | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        pyproject = parent / "pyproject.toml"
        if not pyproject.is_file():
            continue
        match = _VERSION_KEY.search(_project_table(pyproject.read_text(encoding="utf-8")))
        if match:
            return match.group(1)
    return None


def detect_version() -> str:
    try:
        return version(DISTRIBUTION)
    except PackageNotFoundError:
        # Source checkout without an installed distribution.
        return _source_checkout_version() or "0.0.0+unknown"


__version__ = detect_version()
