"""Fallback contract: the zero-dependency path and the learned path (round 6).

The lightweight local path is the zero-dependency fallback the CLI uses when the
optional ML extra is missing, and it is also the path a deployment explicitly
pins with ``--intent-mode heuristic``. Two things have to hold:

* it decides exactly the same things every time, so the pinned decisions below
  are a contract rather than a snapshot;
* when a learned pipeline's components agree with the structural ones, the full
  learned route emits the identical decision. Where they disagree, the learned
  model wins - that is asserted too, so the equality above cannot pass
  vacuously.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eq_layer import cli
from eq_layer.affect import HeuristicAffect
from eq_layer.intent import HeuristicIntent
from eq_layer.policies import Selector
from eq_layer.response_experiment import FullEQPipeline
from eq_layer.tracker import ConversationTracker
from eq_layer.trained_subtext import SubtextDecision

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
FALLBACK_GOLDEN: tuple[dict, ...] = tuple(
    json.loads(line)
    for line in (FIXTURE_DIR / "fallback_golden.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
)

AFFECT_ARTIFACT = "artifacts/affect.joblib"
SUBTEXT_ARTIFACT = "artifacts/subtext.joblib"


class MirroringAffect:
    """Learned affect adapter that reproduces the structural decision exactly."""

    name = "stub-agreeing-affect"

    def infer(self, messages, turn_index=0, annotated=None):
        return HeuristicAffect().infer(messages, turn_index=turn_index, annotated=annotated)


class MirroringSubtext:
    """Learned subtext adapter that reproduces the structural decision exactly."""

    name = "stub-agreeing-subtext"
    structural = HeuristicAffect()

    def infer(self, messages, affect, annotated=None):
        annotated = annotated or {}
        user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
        has_prior_assistant = any(m.get("role") == "assistant" for m in messages[:-1])
        label = annotated.get("subtext") or self.structural._subtext(
            user_turns, {}, has_prior_assistant=has_prior_assistant
        )
        return SubtextDecision(
            label=str(label),
            confidence=1.0,
            signals=None,
            evidence_level="structural",
            evidence=("stub:agrees_with_structural_fallback",),
        )


class MirroringIntent:
    """Learned intent adapter that reproduces the heuristic decision exactly."""

    name = "stub-agreeing-intent"
    fallback = HeuristicIntent()

    def infer(self, messages):
        return self.fallback.infer(messages)

    def reconcile_with_dialogue_evidence(self, state, messages, signals):
        return state


class DisagreeingSubtext(MirroringSubtext):
    """Learned subtext adapter that refuses the structural correction decision."""

    name = "stub-disagreeing-subtext"

    def infer(self, messages, affect, annotated=None):
        annotated = annotated or {}
        user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
        structural = self.structural._subtext(
            user_turns,
            {},
            has_prior_assistant=any(m.get("role") == "assistant" for m in messages[:-1]),
        )
        if annotated.get("subtext"):
            label = str(annotated["subtext"])
        elif structural == "statement":
            label = "correction"
        else:
            label = structural
        return SubtextDecision(
            label=label,
            confidence=1.0,
            signals=None,
            evidence_level="derived",
            evidence=("stub:disagrees_with_structural_fallback",),
        )


def _pipeline(subtext: object | None = None) -> FullEQPipeline:
    return FullEQPipeline(
        tracker=ConversationTracker(
            affect=MirroringAffect(), subtext=subtext or MirroringSubtext()
        ),
        intent=MirroringIntent(),
        selector=Selector(),
    )


@pytest.fixture(autouse=True)
def patched_pipeline_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """The learned artifacts are stubs here, never the real joblib files."""
    monkeypatch.setattr(
        FullEQPipeline, "load", classmethod(lambda cls, affect, subtext, **kwargs: _pipeline())
    )
    # These fixtures explicitly test the absent-ML fallback even on hosts
    # where real scikit-learn is installed for other independent test cases.
    from eq_layer._deps import OptionalDependencyError
    from eq_layer.trained_intent import TrainedIntent

    def unavailable(cls, *args, **kwargs):
        raise OptionalDependencyError("fallback-only reference fixture")

    monkeypatch.setattr(TrainedIntent, "from_bundled", classmethod(unavailable))


@pytest.fixture()
def agreeing_pipeline(patched_pipeline_load: None) -> FullEQPipeline:
    return _pipeline()


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _decision_view(metadata: dict) -> dict:
    """The learned route's metadata reduced to the shared decision fields."""
    return _plain(
        {
            "policy": metadata["policy"],
            "register": metadata["register"],
            "subtext": metadata["subtext"],
            "stance": metadata["stance"],
            "repair": metadata["repair"],
            "interaction_quality": metadata["interaction_quality"],
            "dialogue_reference": {
                key: value
                for key, value in metadata["dialogue_reference"].items()
                if key != "candidate_stack"
            },
            "intent_kind": metadata["intent_kind"],
        }
    )


def _fallback_view(result: dict) -> dict:
    return _plain(
        {
            "policy": result["policy"]["name"],
            "register": result["policy"]["register"],
            "subtext": result["affect"]["subtext"],
            "stance": result["affect"]["stance"],
            "repair": result["repair"],
            "interaction_quality": result["interaction_quality"],
            "dialogue_reference": {
                key: value for key, value in result["reference"].items() if key != "candidate_stack"
            },
            "intent_kind": result["intent"]["kind"],
        }
    )


def _full_route(messages: list[dict]) -> dict:
    return cli._full_route(messages, AFFECT_ARTIFACT, SUBTEXT_ARTIFACT)


# --- the pinned fallback decisions ----------------------------------------


@pytest.mark.parametrize("row", FALLBACK_GOLDEN, ids=lambda row: row["id"])
def test_fallback_decision_matches_the_pinned_object(row: dict) -> None:
    assert _plain(cli._lightweight_route(row["messages"], "heuristic")) == _plain(
        row["fallback"]
    )


@pytest.mark.parametrize("row", FALLBACK_GOLDEN, ids=lambda row: row["id"])
def test_full_route_decision_matches_the_pinned_object(row: dict) -> None:
    assert _full_route(row["messages"]) == _plain(row["routed"])


def test_the_table_pins_at_least_ten_message_sets() -> None:
    assert len(FALLBACK_GOLDEN) >= 10
    pinned = {
        json.dumps(row["messages"], ensure_ascii=True, sort_keys=True) for row in FALLBACK_GOLDEN
    }
    assert len(pinned) == len(FALLBACK_GOLDEN)


@pytest.mark.parametrize("row", FALLBACK_GOLDEN, ids=lambda row: row["id"])
def test_the_fallback_declares_that_it_is_the_fallback(row: dict) -> None:
    result = cli._lightweight_route(row["messages"], "heuristic")
    assert result["runtime_mode"] == "lightweight-local"
    assert result["intent_backend"] == "heuristic"
    # The fallback must not claim learned artifacts it does not use.
    assert "not the full learned" in result["claim_boundary"]
    assert "trained" not in json.dumps(result, ensure_ascii=True)


# --- differential tests ----------------------------------------------------


@pytest.mark.parametrize("row", FALLBACK_GOLDEN, ids=lambda row: row["id"])
def test_the_learned_route_matches_the_fallback_when_the_model_agrees(
    row: dict, agreeing_pipeline: FullEQPipeline
) -> None:
    routed = _full_route(row["messages"])
    fallback = cli._lightweight_route(row["messages"], "heuristic")
    assert _decision_view(routed["metadata"]) == _fallback_view(fallback)
    assert routed["metadata"]["factored_action"] == row["fallback"]["factored_action"]
    assert routed["system_instruction"] == row["fallback"]["system_instruction"]


@pytest.mark.parametrize("row", FALLBACK_GOLDEN, ids=lambda row: row["id"])
def test_auto_and_heuristic_modes_decide_identically(row: dict) -> None:
    """With the ML extra absent, ``auto`` is the fallback, not a different one."""
    auto = cli._lightweight_route(row["messages"], "auto")
    heuristic = cli._lightweight_route(row["messages"], "heuristic")
    differing = {
        key
        for key in set(auto).union(heuristic)
        if _plain(auto.get(key)) != _plain(heuristic.get(key))
    }
    assert differing == {"intent_backend"}
    assert auto["intent_backend"] == "heuristic-fallback-no-ml-extra"
    assert heuristic["intent_backend"] == "heuristic"


def test_a_disagreeing_model_overrides_the_fallback(
    monkeypatch: pytest.MonkeyPatch, agreeing_pipeline: FullEQPipeline
) -> None:
    """The equality above must fail as soon as the model says something else.

    A learned subtext that reads ``correction`` where the structural tracker
    reads ``statement`` has to move the decision, not be smoothed over.
    """
    messages = [{"role": "user", "content": "Randevumu üç kez değiştirdiler"}]
    monkeypatch.setattr(
        FullEQPipeline,
        "load",
        classmethod(lambda cls, affect, subtext, **kwargs: _pipeline(DisagreeingSubtext())),
    )
    routed = _full_route(messages)["metadata"]
    fallback = cli._lightweight_route(messages, "heuristic")
    assert routed["subtext"] == "correction"
    assert routed["policy"] == "ask_one_question"
    assert fallback["affect"]["subtext"] == "statement"
    assert fallback["policy"]["name"] == "mirror_specific"
    assert _decision_view(routed) != _fallback_view(fallback)
