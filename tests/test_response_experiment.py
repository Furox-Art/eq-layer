"""Regression tests for the model-command timeout (audit C1).

No network and no real models: the persistent path is exercised with a
sleeping child that must be force-killed at the deadline, and the happy path
uses a tiny newline-JSON echo child.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from eq_layer import ModelCommandTimeout
from eq_layer.response_experiment import CommandModel, _force_kill

SLEEPER_ARGV = (sys.executable, "-c", "import time; time.sleep(300)")

ECHO_ARGV = (
    sys.executable,
    "-c",
    "import json, sys\n"
    "for line in sys.stdin:\n"
    "    if line.strip():\n"
    "        print(json.dumps({'text': 'ack'}), flush=True)\n",
)

MESSAGES = [{"role": "user", "content": "hello"}]


def test_default_timeout_is_grounded() -> None:
    model = CommandModel(command=("unused",), model_id="m")
    assert model.timeout_seconds == 120


def test_persistent_timeout_raises_typed_error_and_kills_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eq_layer import response_experiment

    spawns: list[subprocess.Popen] = []
    real_popen = response_experiment.subprocess.Popen

    class SpyPopen(real_popen):  # type: ignore[type-arg]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            spawns.append(self)

    monkeypatch.setattr(response_experiment.subprocess, "Popen", SpyPopen)
    model = CommandModel(
        command=SLEEPER_ARGV,
        model_id="sleeper",
        timeout_seconds=1,
        persistent=True,
    )
    started = time.monotonic()
    try:
        with pytest.raises(ModelCommandTimeout, match="within 1 seconds"):
            model.generate(MESSAGES, seed=0, condition="eq")
    finally:
        model.close()
    elapsed = time.monotonic() - started

    # The deadline must fire near 1s, never after the child's 300s sleep.
    assert elapsed < 10
    # One retry: the first wedged child is replaced so a single stall cannot
    # discard a long run, and both are force-killed and reaped.
    assert len(spawns) == 2
    assert all(child.poll() is not None for child in spawns)


def test_persistent_timeout_leaves_no_wedged_child_behind() -> None:
    model = CommandModel(
        command=SLEEPER_ARGV,
        model_id="sleeper",
        timeout_seconds=1,
        persistent=True,
    )
    try:
        with pytest.raises(ModelCommandTimeout):
            model.generate(MESSAGES, seed=0, condition="eq")
        # The failed call dropped its wedged child; the next call starts a
        # fresh process and times out on its own rather than reusing a corpse.
        with pytest.raises(ModelCommandTimeout):
            model.generate(MESSAGES, seed=0, condition="eq")
    finally:
        model.close()


def test_persistent_round_trip_still_works() -> None:
    model = CommandModel(
        command=ECHO_ARGV,
        model_id="echo",
        timeout_seconds=30,
        persistent=True,
    )
    try:
        assert model.generate(MESSAGES, seed=0, condition="baseline") == "ack"
        assert model.generate(MESSAGES, seed=1, condition="eq") == "ack"
    finally:
        started = time.monotonic()
        model.close()
        assert time.monotonic() - started < 10


def test_oneshot_timeout_is_the_same_typed_error() -> None:
    model = CommandModel(
        command=SLEEPER_ARGV,
        model_id="sleeper",
        timeout_seconds=1,
        persistent=False,
    )
    started = time.monotonic()
    with pytest.raises(ModelCommandTimeout, match="within 1 seconds"):
        model.generate(MESSAGES, seed=0, condition="eq")
    assert time.monotonic() - started < 10


def test_force_kill_reaps_child() -> None:
    process = subprocess.Popen(list(SLEEPER_ARGV))
    _force_kill(process)
    assert process.poll() is not None
