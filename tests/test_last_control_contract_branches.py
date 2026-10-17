import json

import pytest

from eq_layer import actions, affect, cli
from eq_layer import dialogue_reference as ref
from eq_layer import intent_belief as belief
from eq_layer import interaction as interaction
from eq_layer import response_experiment as exp
from eq_layer.interaction import InteractionQualityState, RepairState
from eq_layer.policies import POLICIES, AffectState


def msg(role, text):
    return {"role": role, "content": text}


def test_reference_without_prior_user_goal_does_not_invent_one():
    result = ref.infer_dialogue_reference(
        [
            msg("assistant", "Completed."),
            msg("user", "continue"),
        ]
    )
    assert result.active and not result.resolved
    assert "no_safe_goal_anchor" in result.evidence


def test_policy_repair_preserves_prior_clarification_and_adjusts_neutral_social_move():
    ask = next(p for p in POLICIES if p.name == "ask_one_question")
    repair = RepairState(active=True)
    assert actions._repair_move(ask, repair) == "clarify_one"
    neutral = next(p for p in POLICIES if p.name == "execute_request")
    quality = InteractionQualityState(
        current=0.4,
        delta=-0.3,
        repeated_failure_count=1,
        unresolved_repair_count=0,
        clarification_count=1,
    )
    action = actions.compose_action(neutral, interaction_quality=quality)
    assert action.social_move == "low_warmth"


def test_blank_affect_data_record_is_not_a_case(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n \n" + json.dumps({"id": "item"}) + "\n")
    assert affect.load_cases(str(path)) == [{"id": "item"}]


def test_direct_observed_belief_fields_do_not_infer_extra_categories():
    posterior = belief.IntentBelief.from_mapping({"statement": 0.2, "question": 0.8})
    assert posterior.runner_up_kind == "statement"


def test_stance_wrong_is_not_inferred_from_affect():
    state = AffectState(
        valence=0,
        arousal=0,
        escalation_delta=0,
        stance="user_wrong",
        subtext="statement",
        turn_index=0,
    )
    assert state.user_is_wrong and not state.user_is_right


def test_unicode_encoding_error_is_an_explicit_cli_validation_failure(monkeypatch):
    # The secure JSON decoder correctly blocks surrogates first; exercise the
    # standalone validation guard using the same impossible text object.
    monkeypatch.setattr(
        cli, "strict_json_loads", lambda raw: [{"role": "user", "content": "\ud800"}]
    )
    with pytest.raises(ValueError, match="valid UTF-8"):
        cli._load_messages("valid-json")


def test_repeated_correction_scanner_handles_no_assistant_and_trivial_replacement():
    assert interaction._extract_replacement("I said: that") == ""
    assert interaction._last_assistant_before([msg("user", "hello")], 1) is None
    assert interaction._correction_events([msg("user", "I said: correct value")]) == []


def test_persistent_process_close_handles_missing_stdin():
    process = type("Child", (), {"stdin": None, "wait": lambda self, timeout=None: 0})()
    model = exp.CommandModel(command=("demo",), model_id="demo", persistent=True)
    model._process = process
    model.close()
    assert model._process is None


def test_nonfinal_experiment_case_validation_handles_multiple_rows():
    rows = [
        {"id": "a", "transcript": [msg("user", "Hello")]},
        {"id": "b", "transcript": [msg("user", "World")]},
    ]
    report = exp.validate_experiment_cases(rows, final=False)
    assert report["ok"] and report["n_cases"] == 2
