"""Hardening round: lexical audit evasion and supply-chain metadata.

The response auditor decides whether a draft broke a hard contract, so a
spelling-level trick must not be able to hide a privacy promise or a stance
admission. These tests pin the folding rules, the word-boundary requirement and
the packaging/CI metadata that the security policy now describes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from eq_layer.auditor import AuditContext, ResponseAuditor
from eq_layer.textnorm import normalise_light, normalise_match

REPO_ROOT = Path(__file__).resolve().parents[1]


def _context(**overrides: object) -> AuditContext:
    values: dict[str, object] = {
        "task_move": "execute_request",
        "social_move": "neutral",
        "repair_move": "none",
        "verbosity": "normal",
        "question_budget": 0,
        "no_guess": False,
        "stance": "unknown",
    }
    values.update(overrides)
    return AuditContext(**values)


def _codes(response: str, **overrides: object) -> set[str]:
    audit = ResponseAuditor().audit(response, _context(**overrides))
    return {issue.code for issue in audit.issues}


PRIVACY_EVASIONS = (
    "Kimseye-söylemem, tamam mı?",
    "KİMSEYE SÖYLEMEM",
    "ｋｉｍｓｅｙｅ ｓöｙｌｅｍｅｍ",
    "kimseye\u200bsöylemem",
    "This will stay\u00a0between us.",
    "This-will-stay-between-us",
    "I guarantee\nconfidentiality.",
    "completely-confidential",
)

STANCE_EVASIONS = (
    "SEN HAKLISIN!",
    "Sen haklısın, ama kanıt yok.",
    "You’re right.",
    "YOU'RE RIGHT",
    "you are right — this time",
    "I WAS WRONG.",
)


@pytest.mark.parametrize("response", PRIVACY_EVASIONS)
def test_privacy_promise_evasions_are_still_hard_failures(response: str) -> None:
    assert "absolute_privacy_promise" in _codes(response)


@pytest.mark.parametrize("response", STANCE_EVASIONS)
def test_stance_admission_evasions_are_still_hard_failures(response: str) -> None:
    assert "unsupported_stance_admission" in _codes(response)


def test_word_boundaries_do_not_turn_longer_words_into_matches() -> None:
    assert "absolute_privacy_promise" not in _codes(
        "Kimseye söylememek mümkün değil."
    )
    assert "unsupported_stance_admission" not in _codes(
        "You are rightful about that."
    )


def test_matching_fold_is_separate_from_the_light_fold() -> None:
    # The light fold keeps punctuation and case distinctions that the auditor
    # deliberately erases; other callers depend on the unchanged behaviour.
    assert normalise_light("KİMSEYE-SÖYLEMEM") != normalise_match("KİMSEYE-SÖYLEMEM")
    assert normalise_light("A  B") == "a b"
    assert normalise_match("You’re right!") == "you re right"
    assert normalise_match("İ") == "i"
    assert normalise_match("ı") == "i"
    assert normalise_match("kimseye\u200bsöylemem") == "kimseye soylemem"


def test_blank_sentence_fragments_are_ignored() -> None:
    # Leading newlines produce empty fragments; only real sentences may match.
    assert "unsupported_stance_admission" in _codes("\n\nYou're right.")


def test_ml_extra_declares_both_artifact_dependencies() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    ml_extra = re.search(r"^ml = \[(.*?)^\]", pyproject, re.S | re.M)
    assert ml_extra is not None
    assert "joblib>=1.3,<2" in ml_extra.group(1)
    assert "scikit-learn>=1.5,<2" in ml_extra.group(1)
    dev_extra = re.search(r"^dev = \[(.*?)^\]", pyproject, re.S | re.M)
    assert dev_extra is not None
    assert "joblib>=1.3,<2" in dev_extra.group(1)


def test_sdist_ships_the_sources_its_claims_refer_to() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    sdist = pyproject.split("[tool.hatch.build.targets.sdist]", 1)[1]
    for entry in (
        "/tests",
        "/eval",
        "/tools",
        "/docs",
        "/skills",
        "/npm",
        "/.github",
        "/conftest.py",
        "/package.json",
        "/plugin.json",
        "/SECURITY.md",
        "/CHANGELOG.md",
        "/CITATION.cff",
    ):
        assert f'"{entry}"' in sdist


def test_dependabot_covers_actions_and_python() -> None:
    config = (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert "package-ecosystem: github-actions" in config
    assert "package-ecosystem: pip" in config


@pytest.mark.parametrize("name", ["scorecard.yml", "dependency-review.yml"])
def test_new_workflows_pin_every_action_to_a_full_commit_sha(name: str) -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
    uses = [
        line.split("uses:", 1)[1].split("#", 1)[0].strip()
        for line in workflow.splitlines()
        if "uses:" in line
    ]
    assert uses
    for reference in uses:
        assert re.search(r"@[0-9a-f]{40}$", reference), reference


def test_security_policy_documents_the_dataset_downloads() -> None:
    security = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "## Network access" in security
    assert (
        "It reads messages and emits decisions and text; it makes no network calls"
        not in security
    )
    for module in ("coarse_discourse.py", "dbdc3.py", "dialogbank_iso.py"):
        assert module in security
    assert "joblib" in security
