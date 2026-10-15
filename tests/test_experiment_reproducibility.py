"""Experiment reproducibility (hardening round 6).

The recycling harness has to be reproducible: the same experiment, the same
fixture and the same seed must produce byte-identical records, because the
manifest other people read is only meaningful if a rerun cannot silently move a
verdict.

Two sources of drift are therefore quarantined behind injectable seams in
``generate_pairs``:

* control-arm shuffling, behind ``rng`` (seeded from ``seed`` by default);
* the experiment clock, behind ``clock`` (a fixed epoch by default).

The model is a duck-typed stub, so no subprocess, no network and no wall clock
is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

from eq_layer import cli
from eq_layer.response_experiment import (
    DEFAULT_EXPERIMENT_CLOCK_EPOCH,
    experiment_manifest,
    generate_pairs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = REPO_ROOT / "eval" / "cases.jsonl"

CASES = [
    {"id": case["id"], "transcript": case["transcript"]}
    for case in (
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
][:4]

LENGTH_PROFILES = ("one_sentence", "two_sentence", "eq_matched")


class StubModel:
    """Deterministic, in-process stand-in for the model command."""

    model_id = "stub-model-v1"
    temperature = 0.0
    command = ("stub-model", "--deterministic")

    def __init__(self) -> None:
        self.closed = False
        self.requests: list[tuple[int, str]] = []

    def generate(self, messages, *, seed: int, condition: str) -> str:
        self.requests.append((seed, condition))
        last_user = ""
        for message in messages:
            if message.get("role") == "user":
                last_user = str(message.get("content", ""))
        # Two questions on purpose: the auditor flags the question budget for
        # several cases, which exercises the regeneration branch deterministically.
        text = f"Answer for {condition}#{seed}: {last_user}? Is that clear?"
        if condition == "eq_audit_retry":
            text = f"Repaired answer for {condition}#{seed}: {last_user}."
        return text

    def close(self) -> None:
        self.closed = True


class StubPipeline:
    """Duck-typed pipeline standing in for ``FullEQPipeline``.

    The steering itself is the real lightweight decision core, so the records
    carry real factored actions, stance and audit verdicts.
    """

    def steer_messages(self, messages, annotated=None):
        decision = cli._lightweight_route(list(messages), "heuristic")
        metadata = {
            "policy": decision["policy"]["name"],
            "register": decision["policy"]["register"],
            "subtext": decision["affect"]["subtext"],
            "stance": decision["affect"]["stance"],
            "repair": decision["repair"],
            "factored_action": decision["factored_action"],
        }
        steered = [
            {"role": "system", "content": decision["system_instruction"]},
            *messages,
        ]
        return steered, metadata


def _run(**kwargs) -> list[dict]:
    return generate_pairs(
        CASES,
        model=StubModel(),
        pipeline=StubPipeline(),
        seed=13,
        **kwargs,
    )


# --- the pinned reproducibility contract ----------------------------------


def test_the_same_experiment_and_fixture_produce_identical_records() -> None:
    first = _run()
    second = _run()
    assert first == second
    assert len(first) == len(CASES)


def test_the_records_serialise_to_identical_bytes() -> None:
    first = json.dumps(_run(), ensure_ascii=True, sort_keys=True)
    second = json.dumps(_run(), ensure_ascii=True, sort_keys=True)
    assert first == second


def test_the_recorded_verdict_is_reproducible() -> None:
    records = _run()
    assert records
    for record in records:
        audit = record["generation"]["response_audit"]
        initial = record["generation"]["response_audit_initial"]
        assert initial["should_regenerate"] is True
        assert initial["passed"] is False
        codes = {issue["code"] for issue in audit["issues"]}
        assert "question_budget_exceeded" in codes
    assert records == _run()


# --- seeded randomness ----------------------------------------------------


def test_control_arm_order_is_seed_determined() -> None:
    with_profiles = dict(length_profiles=LENGTH_PROFILES)
    first = _run(**with_profiles)
    second = _run(**with_profiles)
    assert [row["generation"]["order"] for row in first] == [
        row["generation"]["order"] for row in second
    ]
    assert any(len(row["generation"]["order"]) > 3 for row in first)


def test_an_injected_generator_reproduces_the_seeded_order() -> None:
    import random as _random

    first = _run(length_profiles=LENGTH_PROFILES)
    injected = generate_pairs(
        CASES,
        model=StubModel(),
        pipeline=StubPipeline(),
        seed=13,
        rng=_random.Random(13),
        length_profiles=LENGTH_PROFILES,
    )
    assert [row["generation"]["order"] for row in injected] == [
        row["generation"]["order"] for row in first
    ]


def test_a_different_seed_changes_the_control_order() -> None:
    """The seed has to reach the shuffling, or the equality above is vacuous."""
    first = generate_pairs(
        CASES,
        model=StubModel(),
        pipeline=StubPipeline(),
        seed=13,
        length_profiles=LENGTH_PROFILES,
    )
    second = generate_pairs(
        CASES,
        model=StubModel(),
        pipeline=StubPipeline(),
        seed=29,
        length_profiles=LENGTH_PROFILES,
    )
    assert [row["generation"]["pair_seed"] for row in first] != [
        row["generation"]["pair_seed"] for row in second
    ]


# --- the injected clock ---------------------------------------------------


def test_the_default_clock_is_a_fixed_epoch() -> None:
    records = _run()
    for record in records:
        assert record["generation"]["generated_at_epoch"] == DEFAULT_EXPERIMENT_CLOCK_EPOCH == 0.0


def test_an_injected_clock_moves_only_the_epoch_field() -> None:
    plain = generate_pairs(CASES, model=StubModel(), pipeline=StubPipeline(), seed=13)
    stamped = generate_pairs(
        CASES, model=StubModel(), pipeline=StubPipeline(), seed=13, clock=lambda: 1735689600.5
    )
    assert plain != stamped
    for before, after in zip(plain, stamped):
        assert after["generation"]["generated_at_epoch"] == 1735689600.5
        before_stripped = {
            "generation": {
                key: value
                for key, value in before["generation"].items()
                if key != "generated_at_epoch"
            },
            **{key: value for key, value in before.items() if key != "generation"},
        }
        after_stripped = {
            "generation": {
                key: value
                for key, value in after["generation"].items()
                if key != "generated_at_epoch"
            },
            **{key: value for key, value in after.items() if key != "generation"},
        }
        assert before_stripped == after_stripped


def test_the_regeneration_branch_is_reproducible() -> None:
    first = _run(auditor_regenerations=1)
    second = _run(auditor_regenerations=1)
    assert first == second
    assert any(record.get("eq_initial") for record in first)
    assert all(record["generation"]["audit_regeneration_count"] == 1 for record in first)


def test_the_length_matched_arms_are_reproducible() -> None:
    with_profiles = dict(length_profiles=LENGTH_PROFILES)
    first = json.dumps(_run(**with_profiles), ensure_ascii=True, sort_keys=True)
    second = json.dumps(_run(**with_profiles), ensure_ascii=True, sort_keys=True)
    assert first == second
    record = json.loads(first)[0]
    for name in LENGTH_PROFILES:
        assert record[f"length_matched_{name}"]
    assert record["length_profiles"] == list(LENGTH_PROFILES)


def test_the_manifest_is_byte_identical_for_the_same_inputs(tmp_path: Path) -> None:
    affect = tmp_path / "affect.joblib"
    subtext = tmp_path / "subtext.joblib"
    affect.write_bytes(b"affect-artifact")
    subtext.write_bytes(b"subtext-artifact")

    def _manifest() -> str:
        return json.dumps(
            experiment_manifest(
                model=StubModel(),
                affect_model=affect,
                subtext_model=subtext,
                cases_path=CASES_PATH,
                global_seed=13,
                n_cases=len(CASES),
                allow_annotations=False,
            ),
            ensure_ascii=True,
            sort_keys=True,
        )

    assert _manifest() == _manifest()
