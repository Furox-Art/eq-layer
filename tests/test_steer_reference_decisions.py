from dataclasses import replace

import pytest

from eq_layer.dialogue_reference import DialogueReferenceState
from eq_layer.intent import HeuristicIntent
from eq_layer.intent_belief import IntentBelief, decide_intent_action
from eq_layer import policies
from eq_layer.interaction import InteractionQualityState, RepairState
from eq_layer.steer import Steer


def _intent():
    return HeuristicIntent().infer([{"role": "user", "content": "Continue?"}])


def test_steering_without_optional_context_keeps_control_action():
    steer = Steer.build(policies.POLICIES[0])
    assert "Control action" in steer.system_instruction()
    assert steer.apply_to_prompt("Text").endswith("User: Text")


def test_steering_includes_risk_model_when_present():
    belief = IntentBelief.from_mapping({"question": 0.7, "statement": 0.3})
    intent = replace(_intent(), belief=belief, risk_decision=decide_intent_action(belief))
    prompt = Steer.build(policies.POLICIES[0], intent).system_instruction()
    assert "Intent belief:" in prompt
    assert "Intent routing decision:" in prompt


@pytest.mark.parametrize(
    ("reference", "fragment"),
    [
        (
            DialogueReferenceState(
                active=True,
                resolved=True,
                form="continue",
                target_excerpt="previous goal",
                anchor_excerpt="earlier task",
            ),
            "Target context anchor",
        ),
        (
            DialogueReferenceState(
                active=True, resolved=True, form="continue", target_excerpt="previous goal"
            ),
            "Resolved reference target",
        ),
        (
            DialogueReferenceState(active=True, resolved=False, form="continue"),
            "No safe structural target",
        ),
    ],
)
def test_reference_instructions_preserve_resolution_status(reference, fragment):
    result = Steer.build(policies.POLICIES[0], _intent(), reference=reference)
    assert fragment in result.system_instruction()


@pytest.mark.parametrize(
    ("repair", "fragment"),
    [
        (
            RepairState(active=True, replacement_explicit=True, replacement_excerpt="new value"),
            "Explicit correction replacement",
        ),
        (RepairState(active=True, replacement_explicit=False), "Ask at most one targeted question"),
        (RepairState(active=False), "Repair state:"),
    ],
)
def test_repair_instructions_preserve_replacement(repair, fragment):
    result = Steer.build(policies.POLICIES[0], _intent(), repair=repair)
    assert fragment in result.system_instruction()


def test_interaction_health_prints_independently():
    quality = InteractionQualityState(
        current=0.7,
        delta=-0.2,
        repeated_failure_count=2,
        unresolved_repair_count=1,
        clarification_count=3,
    )
    result = Steer.build(policies.POLICIES[0], _intent(), interaction_quality=quality)
    assert "Interaction quality:" in result.system_instruction()
