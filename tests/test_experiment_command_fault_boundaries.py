"""Model command timeout and subprocess failure paths are auditable."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from eq_layer import response_experiment as experiment


def _model():
    return experiment.CommandModel(
        command=("demo-cli",),
        model_id="demo",
        timeout_seconds=1,
    )


def test_command_constructor_rejects_blank_command() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        experiment.CommandModel.from_shell("  ", model_id="demo")
    result = experiment.CommandModel.from_shell(
        "python -c 'print(1)'",
        model_id="demo",
        persistent=True,
    )
    assert result.command[0] == "python"
    assert result.persistent


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plaintext", "plaintext"),
        ('{"text":" sample "}', "sample"),
        ('{"response":" other "}', "other"),
    ],
)
def test_json_or_plain_output_is_explicit(raw, expected):
    assert experiment.CommandModel._decode_response(raw) == expected


@pytest.mark.parametrize("raw", ["", " ", "{}", "[]", '{"text":""}', '{"response":null}'])
def test_empty_or_wrong_schema_model_output_does_not_become_success(raw):
    with pytest.raises(RuntimeError, match="empty|JSON must contain"):
        experiment.CommandModel._decode_response(raw)


def test_one_shot_command_returns_actual_model_text(monkeypatch: pytest.MonkeyPatch):
    model = _model()
    calls = []

    def stub(command, **kwargs):
        calls.append((command, json.loads(kwargs["input"])))
        return SimpleNamespace(returncode=0, stdout='{"response":"answered"}', stderr="")

    monkeypatch.setattr(experiment.subprocess, "run", stub)
    assert (
        model.generate([{"role": "user", "content": "hi"}], seed=13, condition="eq") == "answered"
    )
    assert calls[0][1]["seed"] == 13
    assert calls[0][1]["model_id"] == "demo"


def test_nonzero_real_process_exit_is_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=3, stdout="", stderr="fatal"),
    )
    with pytest.raises(RuntimeError, match="failed"):
        _model().generate([], seed=1, condition="baseline")


def test_one_shot_timeout_is_typed(monkeypatch: pytest.MonkeyPatch):
    def expire(*args, **kwargs):
        raise subprocess.TimeoutExpired(["demo-cli"], timeout=1)

    monkeypatch.setattr(experiment.subprocess, "run", expire)
    with pytest.raises(experiment.ModelCommandTimeout, match="did not respond"):
        _model().generate([], seed=1, condition="baseline")


def test_force_kill_covers_escalation_and_benign_os_errors():
    stuck = Mock()
    stuck.wait.side_effect = [subprocess.TimeoutExpired("demo", 5), None]
    experiment._force_kill(stuck)
    stuck.terminate.assert_called_once()
    stuck.kill.assert_called_once()

    dead = Mock()
    dead.terminate.side_effect = OSError("already dead")
    dead.wait.side_effect = [
        subprocess.TimeoutExpired("demo", 5),
        subprocess.TimeoutExpired("demo", 5),
    ]
    dead.kill.side_effect = OSError("already dead")
    experiment._force_kill(dead)
    dead.kill.assert_called_once()


def test_persistent_unavailable_pipes_and_terminated_child_report_errors():
    model = _model()
    model.persistent = True
    model._process = SimpleNamespace(stdin=None, stdout=None)
    with pytest.raises(RuntimeError, match="pipes are unavailable"):
        model._persistent_generate({})

    stdin = Mock()
    stdout = Mock()
    model._process = SimpleNamespace(
        stdin=stdin, stdout=stdout, poll=lambda: 3, stderr=None, returncode=3
    )
    with pytest.raises(RuntimeError, match="exited"):
        model._persistent_generate({})


def test_persistent_empty_reply_is_error(monkeypatch: pytest.MonkeyPatch):
    model = _model()
    model.persistent = True
    pipe = Mock()
    model._process = SimpleNamespace(
        stdin=pipe,
        stdout=Mock(),
        stderr=None,
        poll=lambda: None,
    )
    monkeypatch.setattr(model, "_read_line", lambda *args: "")
    with pytest.raises(RuntimeError, match="empty response"):
        model._persistent_generate({"role": "user"})


def test_reader_without_stdout_fails_and_empty_lines_are_not_invented():
    model = _model()
    with pytest.raises(RuntimeError, match="pipes are unavailable"):
        model._read_line(SimpleNamespace(stdout=None), 1)
    assert model._read_line(SimpleNamespace(stdout=SimpleNamespace(readline=lambda: "")), 1) == ""


def test_close_on_terminated_and_timeout_child():
    model = _model()
    model.close()
    child = Mock()
    model._process = child
    model.close()
    child.stdin.close.assert_called_once()
    assert model._process is None

    child = Mock()
    child.wait.side_effect = [subprocess.TimeoutExpired("demo", 10), None]
    model._process = child
    model.close()
    child.terminate.assert_called_once()
