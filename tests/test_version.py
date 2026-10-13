"""Regression tests for the single-source-of-truth version (audit M4).

The Python side must not carry a hardcoded copy of the version: the only
editable location is `pyproject.toml`, surfaced through
`importlib.metadata` (installed) or a read of the same field (source
checkout).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from eq_layer import __version__, _version, cli

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match is not None
    return match.group(1)


def test_version_matches_pyproject() -> None:
    assert __version__ == _pyproject_version()


def test_source_checkout_fallback_reads_the_same_field() -> None:
    assert _version._source_checkout_version() == _pyproject_version()


def test_detect_version_prefers_installed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_version, "version", lambda _name: "9.9.9")
    assert _version.detect_version() == "9.9.9"


def test_detect_version_falls_back_without_installed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib.metadata import PackageNotFoundError

    def _missing(_name: str) -> str:
        raise PackageNotFoundError(_name)

    monkeypatch.setattr(_version, "version", _missing)
    assert _version.detect_version() == _pyproject_version()


def test_doctor_reports_the_shared_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["doctor"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["package"] == "eq-layer"
    assert payload["version"] == __version__


def test_no_hardcoded_version_in_package_sources() -> None:
    for source in REPO_ROOT.glob("eq_layer/*.py"):
        text = source.read_text(encoding="utf-8")
        assert _pyproject_version() not in text or source.name == "_version.py"


def test_distribution_manifests_share_the_package_version() -> None:
    """Audit D-1: every shipped version field must agree with pyproject."""
    shared = _pyproject_version()

    package = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    assert package["version"] == shared

    for manifest in ("plugin.json", ".codex-plugin/plugin.json"):
        payload = json.loads((REPO_ROOT / manifest).read_text(encoding="utf-8"))
        assert payload["version"] == shared, manifest

    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version:\s*"([^"]+)"', citation)
    assert match is not None, "CITATION.cff carries no version field"
    assert match.group(1) == shared
