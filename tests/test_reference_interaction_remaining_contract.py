from eq_layer import dialogue_reference as ref
from eq_layer import interaction as inter


def msg(role, content):
    return {"role": role, "content": content}


def test_reference_empty_goal_and_transcript_are_unresolved():
    assert ref.infer_dialogue_reference([]).evidence == ("empty_transcript",)
    assert ref.infer_dialogue_reference([msg("assistant", "Hello")]).evidence == ("no_user_turn",)
    assert ref._contains_phrase("", "test") is False
    assert ref._contains_phrase("hello", "") is False
    assert ref._looks_like_user_goal(" ") is False


def test_references_do_not_implicitly_execute_without_an_assistant_offer():
    assert not ref.infer_dialogue_reference([msg("user", "yap")]).resolved
    without_offer = [msg("user", "Bunu yap"), msg("assistant", "Done."), msg("user", "yap")]
    result = ref.infer_dialogue_reference(without_offer)
    assert result.active and not result.resolved
    assert "no_immediate_assistant_offer" in result.evidence


def test_references_correctly_resolve_assistant_offer_and_original_goal():
    cases = [
        (
            [msg("user", "Bunu yap"), msg("assistant", "I can help you"), msg("user", "do it")],
            "assistant",
        ),
        (
            [msg("user", "Bunu yap"), msg("assistant", "Output."), msg("user", "düzelt")],
            "assistant",
        ),
        ([msg("user", "Bunu yap"), msg("assistant", "Done."), msg("user", "devam et")], "user"),
    ]
    for messages, role in cases:
        result = ref.infer_dialogue_reference(messages)
        assert result.resolved and result.target_role == role
        assert result.target_turn_index is not None
    assert not ref.infer_dialogue_reference(
        [msg("user", "Hello"), msg("assistant", "What now?"), msg("user", "continue")]
    ).resolved


def test_references_without_prior_user_goal_cannot_guess_target():
    result = ref.infer_dialogue_reference(
        [msg("user", "Hello"), msg("assistant", "Ready."), msg("user", "continue")]
    )
    assert result.active and not result.resolved
    assert result.requires_clarification


def test_reference_assistant_offer_with_no_prior_user_turn_records_empty_anchor():
    messages = [
        msg("assistant", "I can check the results"),
        msg("user", "do it"),
    ]
    result = ref.infer_dialogue_reference(messages)
    assert result.resolved
    assert result.anchor_excerpt == ""


def test_repair_handles_no_transcript_or_no_user_message():
    assert not inter.infer_repair_state([]).active
    assert not inter.infer_repair_state([msg("assistant", "Hello")]).active
    assert inter.infer_interaction_quality([]).current == 1.0


def test_repair_text_without_explicit_replacement_not_guessed():
    assert inter._clean_replacement("This") == ""
    assert inter._clean_replacement("Corrected value!") == "Corrected value"
    assert inter._extract_replacement("No replacement provided") == ""
    messages = [
        msg("user", "Initial value"),
        msg("assistant", "Original value"),
        msg("user", "That's not what I meant"),
    ]
    result = inter.infer_repair_state(messages)
    assert result.active and result.replacement_explicit is False


def test_repair_repeated_two_turn_corrections_and_recovery():
    messages = [
        msg("user", "First request"),
        msg("assistant", "A"),
        msg("user", "I said: corrected one"),
        msg("assistant", "B"),
        msg("user", "I meant: corrected two"),
    ]
    result = inter.infer_repair_state(messages)
    assert result.active and result.repeated
    assert result.replacement_explicit
    assert result.recent_repair_count == 2
    assert inter.infer_interaction_quality(messages).repeated_failure_count == 1
    messages.append(msg("assistant", "Corrected"))
    assert not inter.infer_repair_state(messages).active


def test_quality_counts_overhead_only_under_visible_repair_pressure():
    messages = [
        msg("user", "Initial request"),
        msg("assistant", "Clarify?"),
        msg("user", "That's not what I meant. I said: amended"),
        msg("assistant", "Do you mean amended?"),
        msg("user", "I meant: updated"),
    ]
    quality = inter.infer_interaction_quality(messages)
    assert quality.clarification_count >= 2
    assert "clarification_overhead_during_repair" in quality.evidence
