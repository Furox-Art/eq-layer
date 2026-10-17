import io
import json
import runpy
import sys

import pytest

from eq_layer import cli
from eq_layer._deps import OptionalDependencyError


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "{}",
        '{"messages":[]}',
        "[1]",
        '[{"role":"user","content":" "}]',
        '[{"role":"tool","content":"bad"}]',
        '[{"role":"user","content":4}]',
        '[{"role":"assistant","content":"Only assistant"}]',
    ],
)
def test_cli_rejects_malformed_transcripts(raw):
    with pytest.raises(ValueError):
        cli._load_messages(raw)


def test_cli_accepts_both_object_and_list_frame():
    record = {"role": "user", "content": "Valid"}
    assert cli._load_messages(json.dumps({"messages": [record]})) == [record]
    assert cli._load_messages(json.dumps([record])) == [record]


def test_cli_empty_stdin_or_unpaired_model_options_fail(capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    with pytest.raises(SystemExit) as err:
        cli.main(["route"])
    assert err.value.code == 2
    with pytest.raises(SystemExit) as err:
        cli.main(
            [
                "route",
                "--messages-json",
                '[{"role":"user","content":"Hi"}]',
                "--affect-model",
                "a.joblib",
            ]
        )
    assert err.value.code == 2


def test_cli_real_learned_route_is_delegated_to_configured_models(monkeypatch, capsys):
    seen = []

    def routed(messages, affect_model, subtext_model):
        seen.append((messages, affect_model, subtext_model))
        return {"mode": "stub-configured-loader"}

    monkeypatch.setattr(cli, "_full_route", routed)
    assert (
        cli.main(
            [
                "route",
                "--messages-json",
                '[{"role":"user","content":"Explain this"}]',
                "--affect-model",
                "affect.joblib",
                "--subtext-model",
                "subtext.joblib",
                "--pretty",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["mode"] == "stub-configured-loader"
    assert seen[0][1:] == ("affect.joblib", "subtext.joblib")


def test_cli_actionable_optional_dependency_failure(monkeypatch, capsys):
    def fail(*args):
        raise OptionalDependencyError("missing extra")

    monkeypatch.setattr(cli, "_full_route", fail)
    assert (
        cli.main(
            [
                "route",
                "--messages-json",
                '[{"role":"user","content":"Hi"}]',
                "--affect-model",
                "a.joblib",
                "--subtext-model",
                "s.joblib",
            ]
        )
        == 2
    )
    assert "pip install" in capsys.readouterr().err


def test_doctor_reports_available_or_missing_ml_without_skipping(monkeypatch, capsys):
    monkeypatch.setattr(cli, "ml_extra_available", lambda: True)
    assert cli.main(["doctor"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ml_extra_available"] is True
    assert "ml_extra_hint" not in result
    monkeypatch.setattr(cli, "ml_extra_available", lambda: False)
    assert cli.main(["doctor"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ml_extra_available"] is False
    assert "ml_extra_hint" in result


def test_cli_module_main_guard_reports_doctor(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["eq-layer", "doctor"])
    with pytest.raises(SystemExit) as err:
        runpy.run_module("eq_layer.cli", run_name="__main__")
    assert err.value.code == 0
    assert json.loads(capsys.readouterr().out)["package"] == "eq-layer"
