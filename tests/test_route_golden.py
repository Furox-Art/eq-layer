"""Golden decision table for the CLI routing paths (hardening round 6).

``eq-layer route`` has three implementations over the same decision core: the
CLI entry point (``cli.main``), the lightweight local path
(``cli._lightweight_route``) and the full learned path
(``cli._full_route``). All of them emit a decision object: selected policy
family, register, compiled factored action and steering instruction, derived
from the same ``Selector``, ``Steer`` and structural-state code.

This module pins those decisions exactly. Any change to a pinned value is a
routing change and has to be deliberate, so the table is a regression guard in
both directions: it fails when a decision moves, and it fails when a decision
family disappears from the table (see the coverage tests).

Determinism is asserted three ways:

* repeated invocation of the same rater produces byte-identical output;
* the selected policy does not depend on the order the policies are declared in
  (there is no first-match rule);
* a fresh interpreter with a different hash seed, locale and encoding still
  produces the identical decision for every table row.

Two annotation-gated policy families cannot be reached from the CLI because the
route entry points do not accept ``annotated`` fields, so they are pinned on the
shared decision core directly.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from eq_layer import cli, policies
from eq_layer._deps import OptionalDependencyError
from eq_layer.affect import HeuristicAffect
from eq_layer.dialogue_reference import infer_dialogue_reference
from eq_layer.intent import HeuristicIntent, IntentState
from eq_layer.interaction import infer_interaction_quality, infer_repair_state
from eq_layer.policies import POLICIES, AffectState, Register, Selector
from eq_layer.steer import Steer

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

ROUTE_GOLDEN: tuple[dict, ...] = tuple(
    json.loads(line)
    for line in (FIXTURE_DIR / "route_golden.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
)

# Families that only an external annotation can reach: the route entry points
# never forward ``annotated`` fields, so a challenge plus an externally supplied
# stance, or an externally supplied drained subtext, cannot happen there.
ANNOTATED_GOLDEN: tuple[dict, ...] = (
    {
        "id": "validate_then_redirect",
        "messages": [
            {"role": "user", "content": "Rehberi değiştirdiler"},
            {"role": "assistant", "content": "Anladım"},
            {"role": "user", "content": "Sen de mi?!"},
        ],
        "annotated": {"stance": "user_right"},
        "policy": "validate_then_redirect",
        "register": "validate_redirect",
        "social_move": "validate_specific",
    },
    {
        "id": "hold_on_exhaustion",
        "messages": [
            {"role": "user", "content": "Sunumu erteledim"},
            {"role": "assistant", "content": "Sorun değil"},
            {"role": "user", "content": "Yoruldum"},
        ],
        "annotated": {"subtext": "exhaustion"},
        "policy": "hold",
        "register": "hold",
        "social_move": "minimal_presence",
    },
    {
        "id": "hold_on_resignation",
        "messages": [
            {"role": "user", "content": "Sunumu erteledim"},
            {"role": "assistant", "content": "Sorun değil"},
            {"role": "user", "content": "Boş verdim"},
        ],
        "annotated": {"subtext": "resignation"},
        "policy": "hold",
        "register": "hold",
        "social_move": "minimal_presence",
    },
)


def _plain(value):
    """Normalise to JSON shapes: tuples become lists across that boundary."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _annotated_route(messages: list[dict], annotated: dict) -> tuple[object, Steer]:
    """The shared decision core with an externally supplied annotation."""
    state = HeuristicAffect().infer(
        messages, turn_index=len(messages), annotated=annotated
    )
    intent = HeuristicIntent().infer(messages)
    selection = Selector().select(state, intent)
    steer = Steer.build(
        selection.policy,
        intent,
        repair=infer_repair_state(messages),
        interaction_quality=infer_interaction_quality(messages),
        reference=infer_dialogue_reference(messages),
    )
    return selection, steer


# --- the table ------------------------------------------------------------


@pytest.mark.parametrize("case", ROUTE_GOLDEN, ids=lambda case: case["id"])
def test_lightweight_route_matches_the_golden_decision(case: dict) -> None:
    assert _plain(cli._lightweight_route(case["messages"], case["intent_mode"])) == _plain(
        case["expected"]
    )


def test_the_table_pins_at_least_fifteen_decision_objects() -> None:
    assert len(ROUTE_GOLDEN) >= 15


def test_the_table_covers_every_reachable_policy_family() -> None:
    pinned = {case["expected"]["policy"]["name"] for case in ROUTE_GOLDEN}
    pinned |= {case["policy"] for case in ANNOTATED_GOLDEN}
    assert pinned == {policy.name for policy in POLICIES}


def test_the_table_covers_every_register_family() -> None:
    pinned = {case["expected"]["policy"]["register"] for case in ROUTE_GOLDEN}
    pinned |= {case["register"] for case in ANNOTATED_GOLDEN}
    assert pinned == {register.value for register in Register}


def test_the_table_covers_every_intent_mode() -> None:
    pinned = {case["intent_mode"] for case in ROUTE_GOLDEN}
    assert pinned == {"auto", "heuristic"}
    # The third CLI mode has no decision to pin without the ML extra: it must
    # fail loudly instead of silently degrading to the heuristic decision.
    with pytest.raises(OptionalDependencyError, match="pip install"):
        cli._lightweight_route(
            [{"role": "user", "content": "Bunu depoya ekle"}], "learned"
        )


def test_the_table_covers_every_intent_kind_the_fallback_can_produce() -> None:
    from eq_layer.intent_belief import INTENT_LABELS

    pinned = {case["expected"]["intent"]["kind"] for case in ROUTE_GOLDEN}
    assert pinned == set(INTENT_LABELS)


@pytest.mark.parametrize("case", ANNOTATED_GOLDEN, ids=lambda case: case["id"])
def test_annotation_gated_families_match_the_golden_decision(case: dict) -> None:
    selection, steer = _annotated_route(case["messages"], case["annotated"])
    assert selection.policy.name == case["policy"]
    assert selection.policy.register.value == case["register"]
    assert steer.action.social_move == case["social_move"]


def test_ambiguity_is_an_error_not_a_silent_first_match() -> None:
    """A tie on specificity and priority must raise, per the README.

    A challenge the caller marked user-right, on top of a sustained escalation
    run, is the one reachable tie: ``validate_then_redirect`` and
    ``hold_sustained_escalation`` are equally constrained.
    """
    messages = [
        {"role": "user", "content": "Neden bu?"},
        {"role": "user", "content": "Neden yine bu sorun?"},
        {"role": "user", "content": "Neden yine rezalet böyle bir hata oldu?"},
        {"role": "user", "content": "Sen de mi?!"},
    ]
    with pytest.raises(ValueError, match="Ambiguous policy selection"):
        _annotated_route(messages, {"stance": "user_right"})


# --- determinism ---------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in ROUTE_GOLDEN
        if case["id"]
        in {
            "ask_one_question_correction",
            "continue_reference_resolved",
            "deflate_tension",
            "hold_sustained_escalation",
        }
    ],
    ids=lambda case: case["id"],
)
def test_cli_entry_point_matches_the_golden_decision(
    case: dict, capsys: pytest.CaptureFixture
) -> None:
    exit_code = cli.main(
        [
            "route",
            "--messages-json",
            json.dumps(case["messages"]),
            "--intent-mode",
            case["intent_mode"],
        ]
    )
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == _plain(case["expected"])


def test_stdin_transcript_matches_the_golden_decision(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    case = next(row for row in ROUTE_GOLDEN if row["id"] == "report_status")
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(case["messages"])))
    exit_code = cli.main(["route", "--intent-mode", case["intent_mode"]])
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == _plain(case["expected"])


def test_repeated_routing_is_byte_identical() -> None:
    first = [
        json.dumps(
            cli._lightweight_route(case["messages"], case["intent_mode"]),
            ensure_ascii=True,
            sort_keys=True,
        )
        for case in ROUTE_GOLDEN
    ]
    # Fresh rater instances and a reversed case order must not move a byte.
    second = [
        json.dumps(
            cli._lightweight_route(case["messages"], case["intent_mode"]),
            ensure_ascii=True,
            sort_keys=True,
        )
        for case in reversed(ROUTE_GOLDEN)
    ]
    assert second == list(reversed(first))


def test_selection_does_not_depend_on_policy_declaration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reordering the policy tuple must not reorder any selection.

    The selector ranks by specificity and then by explicit ``priority``; a
    first-match implementation would follow declaration order and silently
    change decisions when a policy is inserted.
    """
    grid = []
    for subtext in (
        "correction",
        "disclosure_request",
        "exhaustion",
        "resignation",
        "escalating",
        "demand",
        "challenge",
        "question",
        "statement",
    ):
        for stance in ("user_right", "user_wrong", "unknown"):
            for escalating in (True, False):
                for past_n in (True, False):
                    for arousal in (0.2, 0.7):
                        for high_run in (True, False):
                            if past_n:
                                history = tuple(
                                    AffectState(0.0, 0.5, 0.2, "unknown", "", index)
                                    for index in range(4)
                                )
                            elif high_run:
                                history = tuple(
                                    AffectState(0.0, 0.6, 0.0, "unknown", "", index)
                                    for index in range(2)
                                )
                            else:
                                history = ()
                            grid.append(
                                AffectState(
                                    valence=0.5 - arousal,
                                    arousal=arousal,
                                    escalation_delta=0.2 if escalating else -0.1,
                                    stance=stance,
                                    subtext=subtext,
                                    turn_index=1,
                                    history=history,
                                )
                            )
    intents = [
        IntentState(
            kind=kind,
            canonical_request="pinned",
            response_mode="direct",
            confidence=0.9,
            explicit=kind
            in {"action_request", "status_check", "explanation", "question"},
            needs_clarification=clarify,
        )
        for kind in (
            "action_request",
            "status_check",
            "explanation",
            "question",
            "statement",
            "unknown",
        )
        for clarify in (True, False)
    ]

    def _outcome(selector: Selector, state: AffectState, intent: IntentState) -> str:
        try:
            return selector.select(state, intent).policy.name
        except ValueError as exc:
            return f"error:{type(exc).__name__}"

    baseline = Selector()
    expected: dict[tuple[int, str, bool], str] = {}
    for index, state in enumerate(grid):
        for intent in intents:
            expected[(index, intent.kind, intent.needs_clarification)] = _outcome(
                baseline, state, intent
            )

    rng = random.Random(20261014)
    for _ in range(10):
        shuffled = list(POLICIES)
        rng.shuffle(shuffled)
        monkeypatch.setattr(policies, "POLICIES", tuple(shuffled))
        shuffled_selector = Selector()
        for index, state in enumerate(grid):
            for intent in intents:
                key = (index, intent.kind, intent.needs_clarification)
                assert _outcome(shuffled_selector, state, intent) == expected[key]


@pytest.mark.parametrize(
    "environment",
    (
        {"PYTHONHASHSEED": "1", "PYTHONUTF8": "1", "LC_ALL": "tr_TR.UTF-8", "LANG": "tr_TR.UTF-8"},
        {"PYTHONHASHSEED": "987654321", "LC_ALL": "C", "LANG": "C", "PYTHONIOENCODING": "utf-8"},
    ),
    ids=("turkish-locale-utf8", "c-locale-shadow-hash"),
)
def test_a_fresh_interpreter_under_a_different_environment_agrees(
    environment: dict,
) -> None:
    child = (
        "import json,sys;"
        f"sys.path.insert(0, {str(REPO_ROOT)!r});"
        "from eq_layer import cli;"
        "rows=json.load(sys.stdin);"
        "print(json.dumps([cli._lightweight_route(row['messages'], row['intent_mode'])"
        " for row in rows], ensure_ascii=True, sort_keys=True))"
    )
    child_env = dict(os.environ)
    child_env.update(environment)
    completed = subprocess.run(
        [sys.executable, "-c", child],
        input=json.dumps(list(ROUTE_GOLDEN)),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=child_env,
        timeout=90,
        check=True,
    )
    assert json.loads(completed.stdout) == [
        _plain(case["expected"]) for case in ROUTE_GOLDEN
    ]


# --- decision-core hardening ---------------------------------------------


def test_a_blank_user_turn_is_not_an_interrogation_run() -> None:
    """A whitespace-only turn is not a turn that opened with a wh-word.

    Two wh-word turns plus a blank one used to satisfy the three-turn shape, so
    ``_escalating`` reported an interrogation run that never happened.
    """
    messages = [
        {"role": "user", "content": "Neden bu?"},
        {"role": "user", "content": "Neden şimdi?"},
        {"role": "user", "content": "   "},
    ]
    state = HeuristicAffect().infer(messages, turn_index=len(messages))
    assert state.subtext == "statement"
    assert HeuristicAffect()._escalating([m["content"] for m in messages]) is False


def test_a_real_three_turn_wh_run_is_still_an_interrogation_run() -> None:
    """The blank-turn filter must not swallow a genuine run."""
    messages = [
        {"role": "user", "content": "Neden bu?"},
        {"role": "user", "content": "Neden şimdi?"},
        {"role": "user", "content": "Neden yine?"},
    ]
    state = HeuristicAffect().infer(messages, turn_index=len(messages))
    assert state.subtext == "escalating"
    selection, _ = _annotated_route(messages, {})
    assert selection.policy.name == "repair_interrogation"


def test_a_softening_only_turn_floors_arousal_at_zero() -> None:
    """Arousal is a [0, 1] scale; softening must not produce a negative value."""
    state = HeuristicAffect().infer(
        [{"role": "user", "content": "Galiba aslında belki yanılıyorum"}],
        turn_index=1,
    )
    assert state.arousal == 0.0
    assert 0.0 <= state.valence <= 1.0
    for previous in state.history:
        assert 0.0 <= previous.arousal <= 1.0
