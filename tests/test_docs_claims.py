"""Doc-guard: README and skill claims against the code (hardening round 6).

A doc that outlives the code it describes is a hazard, because its claims are
what other people quote. Each test here takes one claim out of the README or the
Agent Skill docs and checks it against the implementation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from eq_layer.affect import HeuristicAffect
from eq_layer.cli import build_parser
from eq_layer.intent import CONSTRAINT_MARKERS, HeuristicIntent
from eq_layer.intent_belief import INTENT_LABELS
from eq_layer import policies
from eq_layer.policies import Register
from eq_layer.tracker import AnnotationStanceResolver
from eq_layer.trained_intent import TrainedIntent

REPO_ROOT = Path(__file__).resolve().parents[1]
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
SKILL = (REPO_ROOT / "skills" / "eq-layer" / "SKILL.md").read_text(encoding="utf-8")
ARCHITECTURE = (
    REPO_ROOT / "skills" / "eq-layer" / "references" / "ARCHITECTURE.md"
).read_text(encoding="utf-8")


def _backticked(text: str) -> set[str]:
    return {token.strip() for token in re.findall(r"`([^`]+)`", text)}


def _line_containing(document: str, needle: str) -> str:
    for line in document.splitlines():
        if needle in line:
            return line
    raise AssertionError(f"No line contains {needle!r}")


def _sentence_containing(document: str, needle: str) -> str:
    """A whitespace-normalised sentence, so wrapped claims read as one."""

    joined = " ".join(document.split())
    for sentence in re.split(r"(?<=\.)\s+", joined):
        if needle in sentence:
            return sentence
    raise AssertionError(f"No sentence contains {needle!r}")


def _section(document: str, heading: str) -> str:
    match = re.search(rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", document, re.S | re.M)
    assert match, f"No section titled {heading!r}"
    return match.group(1)


def _cli_source() -> str:
    from eq_layer import cli as cli_module

    return Path(cli_module.__file__).read_text(encoding="utf-8")


# --- decision families ----------------------------------------------------


def test_readme_policy_families_are_the_real_register_families() -> None:
    claim = _backticked(_line_containing(README, "response-policy family"))
    assert claim == {register.value for register in Register}


def test_readme_intent_labels_are_the_real_intent_labels() -> None:
    (token,) = _backticked(_line_containing(README, "full posterior over"))
    assert set(token.split("/")) == set(INTENT_LABELS)


def test_readme_bundled_corpus_labels_are_the_real_intent_labels() -> None:
    claim = _backticked(_sentence_containing(README, "Turkish and English examples for"))
    assert claim == set(INTENT_LABELS)
    corpus = [
        json.loads(line)
        for line in (REPO_ROOT / "eq_layer" / "data" / "intent_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert {row["label"] for row in corpus} == set(INTENT_LABELS)
    assert len(corpus) >= 30


def test_readme_constraint_names_are_real_constraint_labels() -> None:
    real = {constraint for _, constraint in CONSTRAINT_MARKERS}
    claim = _backticked(_line_containing(README, "Explicit constraints such as"))
    assert claim
    assert claim <= real


def test_readme_evidence_levels_are_levels_the_code_can_emit() -> None:
    bullets = [
        line
        for line in _section(README, "Evidence provenance").splitlines()
        if line.startswith("- ")
    ]
    levels = {token for bullet in bullets for token in _backticked(bullet)}
    assert levels == {"verified", "direct_learned", "derived", "structural", "fallback"}

    source = "".join(
        (REPO_ROOT / "eq_layer" / name).read_text(encoding="utf-8")
        for name in ("trained_subtext.py", "tracker.py")
    )
    for level in levels:
        assert f'"{level}"' in source, f"{level} is documented but never emitted"


def test_readme_annotated_fields_are_the_fields_the_resolver_reads() -> None:
    claim = _backticked(_line_containing(README, "Cases can carry `annotated` fields"))
    assert claim == {"annotated", "stance", "subtext"}


# --- runtime modes --------------------------------------------------------


def test_the_skill_promises_the_two_mode_keys_and_the_cli_emits_them() -> None:
    claim = _backticked(_sentence_containing(SKILL, "reports `runtime_mode`"))
    assert claim == {"runtime_mode", "intent_backend"}

    from eq_layer import cli

    result = cli._lightweight_route([{"role": "user", "content": "Durum ne"}], "heuristic")
    assert result["runtime_mode"] == "lightweight-local"
    assert result["intent_backend"] == "heuristic"


def test_the_cli_emits_only_the_two_documented_runtime_modes() -> None:
    modes = set(re.findall(r'"runtime_mode":\s*"([a-z-]+)"', _cli_source()))
    assert modes == {"lightweight-local", "full-learned"}
    headings = {
        line.lstrip("# ").strip()
        for line in ARCHITECTURE.splitlines()
        if line.startswith("### ")
    }
    assert headings == {"lightweight-local", "full-learned"}


def test_the_cli_intent_modes_match_the_documented_default() -> None:
    route = build_parser()._subparsers._group_actions[0].choices["route"]
    action = next(item for item in route._actions if getattr(item, "dest", None) == "intent_mode")
    assert set(action.choices) == {"auto", "learned", "heuristic"}
    assert action.default == "auto"


def test_the_trained_intent_backend_name_is_the_advertised_one() -> None:
    claim = _backticked(_sentence_containing(ARCHITECTURE, "bundled `TrainedIntent`"))
    assert {"TrainedIntent", "HeuristicIntent"} <= claim
    assert TrainedIntent.name == "trained-tfidf-logreg"
    assert HeuristicIntent().name == "heuristic"


def test_the_affect_backend_name_is_the_advertised_one() -> None:
    claim = _backticked(_sentence_containing(ARCHITECTURE, "affect: `HeuristicAffect`"))
    assert "HeuristicAffect" in claim
    assert HeuristicAffect().name == "heuristic"


def test_the_skill_falls_back_to_the_structural_path_without_artifacts() -> None:
    """The skill promises heuristic affect + structural state without artifacts."""
    line = _sentence_containing(SKILL, "lightweight local path")
    assert "structural dialogue state" in line
    assert "heuristic affect" in line
    assert HeuristicAffect().name == "heuristic"


def test_the_skill_script_is_the_packaged_route_script() -> None:
    script = (
        REPO_ROOT / "skills" / "eq-layer" / "scripts" / "route.py"
    ).read_text(encoding="utf-8")
    assert "from eq_layer.cli import main" in script
    assert 'main(["route"' in script


# --- factored action contract ---------------------------------------------


def test_the_skill_factored_action_keys_are_the_real_ones() -> None:
    from eq_layer.actions import FactoredAction

    claimed = _backticked(_section(SKILL, "Core invariant"))
    assert {"task_move", "social_move", "repair_move"} <= claimed
    real = set(FactoredAction.__dataclass_fields__)
    assert claimed & {"task_move", "social_move", "repair_move"} <= real


def test_the_skill_realization_controls_are_real_controls() -> None:
    from eq_layer import cli

    line = _sentence_containing(SKILL, "controls such as verbosity")
    claimed = (
        "verbosity",
        "directness",
        "warmth",
        "question budget",
        "scope limiting",
        "no-guess",
    )
    for control in claimed:
        assert control in line
    result = cli._lightweight_route(
        [{"role": "user", "content": "Bu raporu kısaca gönder"}], "heuristic"
    )
    assert set(result["factored_action"]["realization"]) == {
        "verbosity",
        "directness",
        "warmth",
        "question_budget",
        "scope_limited",
        "no_guess",
    }


# --- worked examples ------------------------------------------------------


def test_the_question_versus_challenge_example_holds() -> None:
    affect = HeuristicAffect()
    assert affect._subtext(["Sözleşme kaç gün geçerli?"], {}) == "question"
    assert affect._subtext(["Sen de mi?!"], {}) == "challenge"


def test_first_word_matching_strips_punctuation() -> None:
    assert HeuristicAffect()._first_word("Neden?") == "neden"


def test_escalating_past_n_excludes_the_current_turn() -> None:
    """One polite turn after three hostile ones must not reset the register."""
    assert _line_containing(README, "escalating_past_n` excludes the current turn")
    predicate = next(
        policy for policy in policies.POLICIES if policy.name == "hold_sustained_escalation"
    )
    messages = [
        {"role": "user", "content": "Neden bu?"},
        {"role": "user", "content": "Neden yine bu sorun?"},
        {"role": "user", "content": "Neden yine rezalet böyle bir hata oldu?"},
        {"role": "user", "content": "Tamam, sakinim."},
    ]
    state = HeuristicAffect().infer(messages, turn_index=len(messages))
    assert state.escalation_delta <= 0.0
    assert predicate.applies(state, None) is True


def test_stance_is_never_inferred_from_generic_dialogue() -> None:
    messages = [
        {"role": "user", "content": "Sen haklısın, ben yanıldım"},
        {"role": "assistant", "content": "Haklısın"},
        {"role": "user", "content": "Tekrar dene"},
    ]
    assert AnnotationStanceResolver().resolve(messages, annotated={}).label == "unknown"


# --- layout ----------------------------------------------------------------


def test_every_layout_path_mentioned_in_the_readme_exists() -> None:
    block = re.search(r"## Layout\s*\n```\n(.*?)```", README, re.S)
    assert block
    missing = []
    prefix = ""
    for entry in block.group(1).splitlines():
        stripped = entry.rstrip()
        if stripped.endswith("/"):
            prefix = stripped.strip()
            continue
        match = re.match(r"\s*([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)\s+", entry)
        if not match:
            continue
        candidate = REPO_ROOT / prefix.rstrip("/") / match.group(1)
        if not candidate.exists():
            missing.append(str(candidate))
    assert not missing


def test_every_path_referenced_by_the_skill_exists() -> None:
    for needle in _backticked(SKILL):
        if needle.startswith(("scripts/", "references/")):
            assert (REPO_ROOT / "skills" / "eq-layer" / needle).exists(), needle


def test_the_architecture_reference_documents_the_runtime_modes_and_loss_matrix() -> None:
    claim = _backticked(_sentence_containing(ARCHITECTURE, "production loss matrix"))
    assert claim == {"hand-specified-v1"}
    assert "hand-specified-v1" in (REPO_ROOT / "eq_layer" / "intent_belief.py").read_text(
        encoding="utf-8"
    )


def test_the_architecture_control_path_matches_the_route_output() -> None:
    from eq_layer import cli

    path = re.search(r"## Control path\s*\n```text\n(.*?)```", ARCHITECTURE, re.S)
    assert path
    for component in ("affect", "intent", "repair", "interaction-quality", "reference"):
        assert component in path.group(1)

    result = cli._lightweight_route(
        [
            {"role": "user", "content": "Bu tabloyu düzelt"},
            {"role": "assistant", "content": "İstersen düzeltmeyi ben yapabilirim."},
            {"role": "user", "content": "devam"},
        ],
        "heuristic",
    )
    assert set(result) >= {
        "affect",
        "intent",
        "repair",
        "interaction_quality",
        "reference",
        "policy",
        "factored_action",
        "system_instruction",
    }
