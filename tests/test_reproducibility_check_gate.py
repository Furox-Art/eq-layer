"""The reproducibility check must fail the job, not only produce a report."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "eval" / "reproducibility_check.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "eq_reproducibility_check_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(monkeypatch, tmp_path, *, within, across, require):
    module = _load_script()
    monkeypatch.setattr(
        module,
        "persistent_requests",
        lambda command, payload, repeats: list(within),
    )
    answers = iter(across)
    monkeypatch.setattr(module, "request", lambda command, payload: next(answers))
    output = tmp_path / "report.json"
    argv = [
        "reproducibility_check.py",
        "--model-command",
        "stub-model",
        "--output",
        str(output),
    ]
    if require:
        argv.append("--require-reproducible")
    monkeypatch.setattr(sys, "argv", argv)
    return module.main(), json.loads(output.read_text(encoding="utf-8"))


def test_within_process_state_leak_fails_when_required(monkeypatch, tmp_path):
    code, report = _run(
        monkeypatch, tmp_path, within=["a", "b", "a"], across=["a", "a"], require=True
    )
    assert code == 3
    assert report["within_process"]["reproducible"] is False
    assert report["gate"] == {
        "require_reproducible": True,
        "within_process_passed": False,
    }


def test_state_leak_is_only_reported_without_the_flag(monkeypatch, tmp_path):
    code, report = _run(
        monkeypatch, tmp_path, within=["a", "b", "a"], across=["a", "a"], require=False
    )
    assert code == 0
    assert report["within_process"]["reproducible"] is False
    assert report["gate"]["require_reproducible"] is False


def test_identical_within_process_output_passes_the_gate(monkeypatch, tmp_path):
    code, report = _run(
        monkeypatch, tmp_path, within=["a", "a", "a"], across=["a", "a"], require=True
    )
    assert code == 0
    assert report["within_process"]["reproducible"] is True
    assert report["across_process"]["reproducible"] is True
