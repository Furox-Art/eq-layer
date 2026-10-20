from dataclasses import replace

import pytest

from eq_layer.actions import compose_action
from eq_layer.affect import HeuristicAffect
from eq_layer.dialogue_reference import DialogueReferenceState
from eq_layer.intent import HeuristicIntent
from eq_layer.intent_belief import IntentBelief, expected_loss
from eq_layer.interaction import InteractionQualityState, RepairState
from eq_layer import policies
from eq_layer.policies import AffectState, Selector


def intent(text="Bunu yap"):
    return HeuristicIntent().infer([{"role": "user", "content": text}])


def policy():
    return next(p for p in policies.POLICIES if p.name == "mirror_specific")


def test_repair_moves_distinguish_repeated_and_corrected_tasks():
    for repeated, explicit, expected in [
        (True, True, "stop_restatement_and_apply_correction"),
        (False, True, "apply_correction"),
        (True, False, "stop_restatement_and_repair"),
        (False, False, "clarify_one"),
    ]:
        repair = RepairState(active=True, repeated=repeated, replacement_explicit=explicit)
        action = compose_action(policy(), intent(), repair=repair)
        assert action.repair_move == expected
        assert action.realization.question_budget == 1
    assert (
        compose_action(policy(), intent(), repair=RepairState(active=False)).repair_move == "none"
    )


def test_realization_explicit_constraints_override_policy_surface():
    instructions = replace(
        intent("Bunu yap"), constraints=("brief", "detailed", "direct", "scope_limited", "no_guess")
    )
    action = compose_action(policy(), instructions)
    assert action.realization.verbosity == "high"
    assert action.realization.directness == "high"
    assert action.realization.scope_limited is True
    assert action.realization.no_guess is True


def test_low_interaction_quality_reduces_surface_without_replacing_task():
    low = InteractionQualityState(
        current=0.4,
        delta=-0.3,
        repeated_failure_count=3,
        unresolved_repair_count=1,
        clarification_count=1,
    )
    action = compose_action(policy(), intent(), interaction_quality=low)
    assert action.realization.verbosity == "low"
    assert action.realization.warmth == "low"
    assert action.task_move == "execute_request"


def test_reference_and_repair_clarification_require_no_guess():
    for kwargs in [
        {"reference": DialogueReferenceState(active=True, resolved=False)},
        {"repair": RepairState(active=True, replacement_explicit=False)},
    ]:
        action = compose_action(policy(), intent(), **kwargs)
        assert action.realization.no_guess
        assert action.realization.question_budget == 1


def test_affect_empty_turn_and_high_arousal_run_are_independent():
    obj = HeuristicAffect()
    assert obj._arousal("") == 0.0
    assert obj._recent_user_turns(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "b"},
        ]
    ) == ["a", "b"]
    assert not obj._high_arousal_run(["Hello"])
    assert obj._high_arousal_run(["Neden rezalet!", "Neden yine rezalet!"])
    assert obj._demand_pattern(["Yap!", "Yap!", "Yap!"])
    assert not obj._demand_pattern(["Yap!", "Yap!"])


def test_belief_preserves_unknown_and_rejects_unsupported_intent():
    with pytest.raises(ValueError, match="Unsupported intent labels"):
        IntentBelief.from_mapping({"unrecognized": 1.0})
    prior = IntentBelief.from_mapping({})
    assert prior.top_kind == "unknown"
    assert prior.normalized_entropy == 0.0
    assert prior.probability("absent") == 0.0
    with pytest.raises(ValueError, match="Unsupported intent decision"):
        expected_loss(prior, "invented")


def test_selector_explicit_override_is_audited():
    state = AffectState(
        valence=0,
        arousal=0,
        escalation_delta=0,
        stance="unknown",
        subtext="statement",
        turn_index=0,
    )
    chooser = Selector(overrides={"statement": "hold"})
    result = chooser.select(state)
    assert result.policy.name == "hold"
    assert result.rationale == "override:hold"
