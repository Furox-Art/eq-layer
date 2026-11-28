"""Tests verifying first-class Russian language support across EQ-Layer."""

from __future__ import annotations

import pytest

from eq_layer.affect import HeuristicAffect
from eq_layer.auditor import AuditContext, ResponseAuditor
from eq_layer.dialogue_reference import infer_dialogue_reference
from eq_layer.intent import HeuristicIntent
from eq_layer.steer import Steer, GENERIC_PHRASES
from eq_layer.textnorm import normalise_light, normalise_plain, normalise_match
from eq_layer.policies import REGISTRY, Selector
from eq_layer.actions import compose_action

try:
    from eq_layer.trained_intent import TrainedIntent
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def test_russian_textnorm_yo_folding() -> None:
    assert normalise_light("Переделай ВСЁ пожалуйста") == "переделай все пожалуйста"
    assert normalise_plain("Ещё раз: исправь ошибку!") == "еще раз исправь ошибку"
    assert "еще раз" in normalise_match("ЕЩЁ РАЗ")


@pytest.mark.parametrize(
    ("text", "expected_kind", "expected_mode"),
    [
        ("Добавь эту функцию в модуль", "action_request", "execute"),
        ("Исправь ошибку в коде", "action_request", "execute"),
        ("Какой сейчас статус задачи?", "status_check", "direct"),
        ("Что по статусу деплоя", "status_check", "direct"),
        ("Почему падает тест?", "explanation", "explain"),
        ("Как работает этот алгоритм?", "explanation", "explain"),
        ("Сколько памяти занимает процесс?", "question", "direct"),
        ("Это", "unknown", "clarify"),
        ("Штука", "unknown", "clarify"),
        ("Сегодня сервер работает стабильно", "statement", "contextual"),
    ],
)
def test_russian_heuristic_intent_classification(text: str, expected_kind: str, expected_mode: str) -> None:
    intent = HeuristicIntent().infer([{"role": "user", "content": text}])
    assert intent.kind == expected_kind
    assert intent.response_mode == expected_mode


def test_russian_constraint_detection() -> None:
    intent = HeuristicIntent().infer([{"role": "user", "content": "Исправь баг кратко и четко, без догадок"}])
    assert intent.kind == "action_request"
    assert "brief" in intent.constraints
    assert "direct" in intent.constraints
    assert "no_guess" in intent.constraints


def test_russian_affect_detection() -> None:
    affect_engine = HeuristicAffect()

    # Escalation
    state_calm = affect_engine.infer([{"role": "user", "content": "Привет"}])
    state_angry = affect_engine.infer([{"role": "user", "content": "Ты опять не понимаешь, это меня бесит!"}])
    assert state_angry.arousal > state_calm.arousal

    # Subtext: correction
    messages_correction = [
        {"role": "user", "content": "Сделай таблицу"},
        {"role": "assistant", "content": "Вот график."},
        {"role": "user", "content": "Нет, я не это имел в виду, сделай таблицу."},
    ]
    tracked = affect_engine.infer(messages_correction)
    assert tracked.subtext == "correction"

    # Subtext: disclosure_request
    messages_secret = [
        {"role": "user", "content": "Сохрани это в тайне, это секрет."},
    ]
    tracked_secret = affect_engine.infer(messages_secret)
    assert tracked_secret.subtext == "disclosure_request"


def test_russian_dialogue_reference_continuation() -> None:
    messages = [
        {"role": "user", "content": "Запусти тесты"},
        {"role": "assistant", "content": "Первая часть завершена."},
        {"role": "user", "content": "продолжай"},
    ]
    ref = infer_dialogue_reference(messages)
    assert ref.active is True
    assert ref.resolved is True
    assert ref.form == "continue"
    assert ref.target_turn_index == 0


def test_russian_dialogue_reference_offer_acceptance() -> None:
    messages = [
        {"role": "user", "content": "Что дальше?"},
        {"role": "assistant", "content": "Могу сделать рефакторинг этого метода."},
        {"role": "user", "content": "сделай это"},
    ]
    ref = infer_dialogue_reference(messages)
    assert ref.active is True
    assert ref.resolved is True
    assert ref.form == "execute_reference"
    assert ref.target_turn_index == 1
    assert "рефакторинг" in ref.target_excerpt


def test_russian_auditor_flags_absolute_privacy() -> None:
    auditor = ResponseAuditor()
    ctx = AuditContext(
        task_move="execute",
        social_move="neutral",
        repair_move="none",
        verbosity="normal",
        question_budget=1,
        no_guess=False,
    )
    audit = auditor.audit("Не волнуйтесь, полная конфиденциальность, это останется между нами.", ctx)
    assert audit.passed is False
    codes = [issue.code for issue in audit.issues]
    assert "absolute_privacy_promise" in codes


def test_russian_auditor_flags_stance_admission_when_unknown() -> None:
    auditor = ResponseAuditor()
    ctx = AuditContext(
        task_move="execute",
        social_move="neutral",
        repair_move="none",
        verbosity="normal",
        question_budget=1,
        no_guess=False,
        stance="unknown",
    )
    audit = auditor.audit("Ты прав, я ошибся в предыдущем шаге.", ctx)
    assert audit.passed is False
    codes = [issue.code for issue in audit.issues]
    assert "unsupported_stance_admission" in codes


def test_russian_generic_phrases_included_in_steer() -> None:
    assert any("я понимаю" in phrase for phrase in GENERIC_PHRASES)
    assert any("мне жаль" in phrase for phrase in GENERIC_PHRASES)


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn required for TrainedIntent")
def test_russian_trained_intent_inference() -> None:
    clf = TrainedIntent.from_bundled()
    res = clf.infer([{"role": "user", "content": "Почини падение сервера при запуске"}])
    assert res.kind == "action_request"
    assert res.confidence > 0.3
