"""Offline latency + quality A/B benchmark: EQ layer enabled vs disabled.

The layer is deterministic, so the honest question is not "does it help" but
"what does it cost and what does it change". This harness answers both on a
fixed set of user requests:

  - added latency per request, with classification time and routing time
    measured independently rather than as one lumped number, and
    classification further split into intent inference vs affect/dialogue
    state so the expensive half is visible;
  - quality on script-computed metrics: intent-label accuracy against the
    labelled expected intents, an appropriateness rubric, and a
    communication-quality score.

There is no LLM in the measured path. The EQ layer emits a control decision
(task_move / social_move / repair_move / realization) that is injected into
the decode path; producing the tokens is the model's job and is not part of
what this layer pays for. What is measured here is the layer's own compute
plus the structural quality of the decision it produces. When the optional
learned adapters (scikit-learn) are installed, the learned path is measured;
the zero-dependency heuristic path is measured as well, so the benchmark runs
with or without the ML extra.

Quality is scored on the control decision, not on generated prose, because
the harness has no model to generate prose. The labels a layer-off request
receives are the empty/unlabelled defaults, so every arm is scored through the
same script and the deltas are real measured differences in the decision the
layer makes.

Usage:
    python eval/ab_latency_eval.py
    python eval/ab_latency_eval.py --runs 200 --output eval/results/ab-latency.json

Exit code is non-zero if any measured invariant fails.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.actions import compose_action  # noqa: E402
from eq_layer.affect import HeuristicAffect  # noqa: E402
from eq_layer.dialogue_reference import infer_dialogue_reference  # noqa: E402
from eq_layer.intent import HeuristicIntent  # noqa: E402
from eq_layer.policies import Selector  # noqa: E402
from eq_layer.tracker import (  # noqa: E402
    infer_interaction_quality,
    infer_repair_state,
)

DEFAULT_CASES = Path(__file__).with_name("cases.jsonl")
DEFAULT_HOLDOUT = Path(__file__).with_name("intent_holdout.jsonl")
DEFAULT_RESULTS = Path(__file__).parent / "results" / "ab-latency.json"


# --------------------------------------------------------------------------
# Metrics defined up front, before the run, so the numbers are not chosen
# after seeing them.
# --------------------------------------------------------------------------

# Appropriateness rubric: 8 structural properties a control decision should
# carry for the conversation state it was computed from. Each is a boolean
# check over the decision, not a judgement of prose.
APPROPRIATENESS_CHECKS = (
    "task_move_present",
    "social_move_present",
    "repair_move_present",
    "realization_present",
    "question_budget_within_limit",
    "no_guess_when_clarifying",
    "boundary_keeps_warmth_low",
    "register_is_known",
)

# Communication-quality rubric: 6 properties of the surface controls.
COMMUNICATION_CHECKS = (
    "verbosity_is_low_or_normal",
    "directness_is_high_or_normal",
    "warmth_is_low_or_normal",
    "question_budget_is_boolean_consistent",
    "scope_limited_is_boolean",
    "no_guess_is_boolean",
)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Deterministic, no interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


@dataclass
class ArmResult:
    """One measured pass over the case set for one arm."""

    total_ms: list[float]
    classification_ms: list[float]
    routing_ms: list[float]
    intent_ms: list[float]
    affect_ms: list[float]
    intent_correct: list[int]
    appropriateness: list[float]
    communication: list[float]


def summarise(values: list[float]) -> dict:
    return {
        "mean_ms": round(statistics.fmean(values), 4) if values else 0.0,
        "median_ms": round(statistics.median(values), 4) if values else 0.0,
        "p50_ms": round(percentile(values, 50), 4),
        "p95_ms": round(percentile(values, 95), 4),
        "min_ms": round(min(values), 4) if values else 0.0,
        "max_ms": round(max(values), 4) if values else 0.0,
        "n": len(values),
    }


# --------------------------------------------------------------------------
# The two arms.
# --------------------------------------------------------------------------


def run_layer_on(messages: list[dict], intent, affect) -> tuple[dict, dict]:
    """Full EQ control path. Returns (decision, timing_ms).

    Classification is timed as a whole and also split into its two halves so
    the report can say which half the cost lives in without a second run.
    """
    t_total0 = time.perf_counter()

    t0 = time.perf_counter()
    tracked = affect.infer(messages, turn_index=len(messages))
    repair = infer_repair_state(messages)
    interaction = infer_interaction_quality(messages)
    reference = infer_dialogue_reference(messages)
    t_affect = time.perf_counter() - t0

    t1 = time.perf_counter()
    intent_state = intent.infer(messages)
    t_intent = time.perf_counter() - t1

    t_class = t_affect + t_intent

    t2 = time.perf_counter()
    selection = Selector().select(tracked, intent_state)
    action = compose_action(
        selection.policy,
        intent_state,
        repair=repair,
        interaction_quality=interaction,
        reference=reference,
    )
    t_route = time.perf_counter() - t2

    total = time.perf_counter() - t_total0
    decision = {
        "policy": selection.policy.name,
        "register": selection.policy.register.value,
        "task_move": action.task_move,
        "social_move": action.social_move,
        "repair_move": action.repair_move,
        "realization": action.realization.__dict__,
        "intent_kind": intent_state.kind,
        "needs_clarification": intent_state.needs_clarification,
    }
    return decision, {
        "classification_ms": t_class * 1000.0,
        "routing_ms": t_route * 1000.0,
        "total_ms": total * 1000.0,
        "intent_ms": t_intent * 1000.0,
        "affect_ms": t_affect * 1000.0,
    }


def run_layer_off(messages: list[dict]) -> tuple[dict, dict]:
    """No control layer. The request is sent to the model as-is, so the
    'decision' is the empty default: no policy, no task move, no surface
    controls. The cost measured here is the transcript handoff a caller
    performs in both arms, so it cancels in the added-latency delta.
    """
    t0 = time.perf_counter()
    # Normalise the transcript the way a caller would before dispatch. This is
    # identical in both arms, so it cancels in the delta.
    payload = [dict(m) for m in messages]
    del payload
    decision = {
        "policy": None,
        "register": None,
        "task_move": None,
        "social_move": None,
        "repair_move": None,
        "realization": None,
        "intent_kind": None,
        "needs_clarification": False,
    }
    total = time.perf_counter() - t0
    return decision, {
        "classification_ms": 0.0,
        "routing_ms": 0.0,
        "total_ms": total * 1000.0,
        "intent_ms": 0.0,
        "affect_ms": 0.0,
    }


# --------------------------------------------------------------------------
# Quality scoring. Same script for both arms.
# --------------------------------------------------------------------------


def score_intent_accuracy(decision: dict, expected: str | None) -> int:
    if not expected:
        return 0
    return int(decision.get("intent_kind") == expected)


def score_appropriateness(decision: dict) -> float:
    real = decision.get("realization")
    budget = int(real.get("question_budget", -1)) if real else -1
    is_boundary = decision.get("policy") == "boundary"
    checks = {
        "task_move_present": bool(decision.get("task_move")),
        "social_move_present": bool(decision.get("social_move")),
        "repair_move_present": decision.get("repair_move") is not None,
        "realization_present": real is not None,
        "question_budget_within_limit": 0 <= budget <= 1,
        "no_guess_when_clarifying": (
            real is not None
            and (not decision.get("needs_clarification") or real.get("no_guess"))
        ),
        # A boundary response must not arrive warm: warmth is what turns a
        # limit into a moral judgement. The policy table already pins BOUNDARY
        # to warmth=low, so this check verifies the wiring, not a preference.
        "boundary_keeps_warmth_low": (
            (not is_boundary) or (real is not None and real.get("warmth") == "low")
        ),
        "register_is_known": decision.get("register") is not None,
    }
    return round(sum(1 for v in checks.values() if v) / len(APPROPRIATENESS_CHECKS), 4)


def score_communication(decision: dict) -> float:
    real = decision.get("realization")
    budget = int(real.get("question_budget", 0)) if real else 0
    checks = {
        "verbosity_is_low_or_normal": real is not None
        and real.get("verbosity") in {"low", "normal"},
        "directness_is_high_or_normal": real is not None
        and real.get("directness") in {"high", "normal"},
        "warmth_is_low_or_normal": real is not None
        and real.get("warmth") in {"low", "normal"},
        "question_budget_is_boolean_consistent": 0 <= budget <= 1,
        "scope_limited_is_boolean": real is not None
        and isinstance(real.get("scope_limited"), bool),
        "no_guess_is_boolean": real is not None
        and isinstance(real.get("no_guess"), bool),
    }
    return round(sum(1 for v in checks.values() if v) / len(COMMUNICATION_CHECKS), 4)


# --------------------------------------------------------------------------
# Benchmark loop.
# --------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def measure_pass(cases: list[dict], *, on: bool, intent, affect) -> ArmResult:
    result = ArmResult([], [], [], [], [], [], [], [])
    for case in cases:
        messages = case["transcript"]
        if on:
            decision, timing = run_layer_on(messages, intent, affect)
        else:
            decision, timing = run_layer_off(messages)
        result.total_ms.append(timing["total_ms"])
        result.classification_ms.append(timing["classification_ms"])
        result.routing_ms.append(timing["routing_ms"])
        result.intent_ms.append(timing["intent_ms"])
        result.affect_ms.append(timing["affect_ms"])
        result.intent_correct.append(
            score_intent_accuracy(decision, case.get("expected_intent"))
        )
        result.appropriateness.append(score_appropriateness(decision))
        result.communication.append(score_communication(decision))
    return result


def run_adapter(
    name: str,
    intent,
    *,
    cases: list[dict],
    holdout: list[dict],
    runs: int,
) -> dict:
    """Measure one intent adapter in both arms."""
    affect = HeuristicAffect()

    # Warm both paths so first-call costs (vectorizer state, allocation) do not
    # land in the reported latencies.
    measure_pass(cases[:2], on=True, intent=intent, affect=affect)
    measure_pass(cases[:2], on=False, intent=intent, affect=affect)

    on_total: list[float] = []
    off_total: list[float] = []
    on_class: list[float] = []
    on_route: list[float] = []
    on_intent: list[float] = []
    on_affect: list[float] = []
    per_case_on: dict[str, list[float]] = {}

    for _ in range(runs):
        res_on = measure_pass(cases, on=True, intent=intent, affect=affect)
        res_off = measure_pass(cases, on=False, intent=intent, affect=affect)
        on_total.extend(res_on.total_ms)
        off_total.extend(res_off.total_ms)
        on_class.extend(res_on.classification_ms)
        on_route.extend(res_on.routing_ms)
        on_intent.extend(res_on.intent_ms)
        on_affect.extend(res_on.affect_ms)
        for case, value in zip(cases, res_on.total_ms):
            per_case_on.setdefault(case["id"], []).append(value)

    # Quality is computed once per case on the final decision; it does not vary
    # across timing runs because the layer is deterministic.
    final_on = measure_pass(cases, on=True, intent=intent, affect=affect)
    final_off = measure_pass(cases, on=False, intent=intent, affect=affect)

    labelled = [c for c in cases if c.get("expected_intent")]
    n_labelled = len(labelled)
    on_correct = sum(final_on.intent_correct)
    off_correct = sum(final_off.intent_correct)

    # The holdout file has no transcripts, only one-turn text plus an expected
    # intent label, so it is scored as its own single-turn request set.
    holdout_correct = 0
    for row in holdout:
        state = intent.infer([{"role": "user", "content": row["text"]}])
        if state.kind == row["label"]:
            holdout_correct += 1

    added = [a - b for a, b in zip(on_total, off_total)]

    return {
        "intent_adapter": name,
        "latency_ms": {
            "layer_on": summarise(on_total),
            "layer_off": summarise(off_total),
            "added": summarise(added),
            "classification_on": summarise(on_class),
            "routing_on": summarise(on_route),
            "classification_intent_on": summarise(on_intent),
            "classification_affect_on": summarise(on_affect),
        },
        "quality": {
            "intent_label_accuracy": {
                "n_labelled": n_labelled,
                "layer_on": round(on_correct / n_labelled, 4) if n_labelled else None,
                "layer_off": (
                    round(off_correct / n_labelled, 4) if n_labelled else None
                ),
                "delta": (
                    round((on_correct - off_correct) / n_labelled, 4)
                    if n_labelled
                    else None
                ),
            },
            "intent_holdout_accuracy": {
                "n": len(holdout),
                "layer_on": (
                    round(holdout_correct / len(holdout), 4) if holdout else None
                ),
            },
            "appropriateness_rubric": {
                "n_cases": len(cases),
                "layer_on": round(statistics.fmean(final_on.appropriateness), 4),
                "layer_off": round(statistics.fmean(final_off.appropriateness), 4),
                "delta": round(
                    statistics.fmean(final_on.appropriateness)
                    - statistics.fmean(final_off.appropriateness),
                    4,
                ),
            },
            "communication_quality": {
                "n_cases": len(cases),
                "layer_on": round(statistics.fmean(final_on.communication), 4),
                "layer_off": round(statistics.fmean(final_off.communication), 4),
                "delta": round(
                    statistics.fmean(final_on.communication)
                    - statistics.fmean(final_off.communication),
                    4,
                ),
            },
        },
        "per_case_added_ms": {
            case_id: round(statistics.fmean(values), 4)
            for case_id, values in per_case_on.items()
        },
    }


def learned_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


def run_benchmark(
    cases: list[dict],
    holdout: list[dict],
    *,
    runs: int,
) -> dict:
    adapters: list[tuple[str, object]] = [
        ("heuristic-zero-dependency", HeuristicIntent())
    ]
    learned = learned_available()
    if learned:
        from eq_layer.trained_intent import TrainedIntent

        adapters.insert(
            0, ("trained-tfidf-logreg-bundled", TrainedIntent.from_bundled())
        )

    arms = {
        name: run_adapter(name, intent, cases=cases, holdout=holdout, runs=runs)
        for name, intent in adapters
    }

    return {
        "benchmark": "eq-layer-ab-latency-quality",
        "mode": {
            "llm_in_measured_path": False,
            "learned_adapters_available": learned,
            "affect_adapter": "heuristic structural (no EmoBank artifact bundled)",
            "note": (
                "No LLM call is measured. The layer emits a control decision "
                "that is injected into the decode path; token generation is "
                "the model's cost, not the layer's. The added latency is the "
                "layer's own deterministic compute."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "design": {
            "unit": "one pass over the full case set",
            "n_cases": len(cases),
            "n_runs_per_arm": runs,
            "n_measurements_per_arm": len(cases) * runs,
            "same_requests_both_arms": True,
            "fixed_seed": "layer is deterministic; no sampling",
            "warmup_passes": 2,
            "classification_measured_independently_of_routing": True,
            "classification_split": (
                "intent inference vs affect/repair/interaction/reference state, "
                "timed as two separate spans inside one classification call"
            ),
            "request_sets": {
                "cases.jsonl": f"{len(cases)} seeded development cases, "
                f"{len([c for c in cases if c.get('expected_intent')])} with an "
                "expected_intent label",
                "intent_holdout.jsonl": (
                    f"{len(holdout)} single-turn holdout requests, all with an "
                    "expected intent label"
                ),
            },
        },
        "arms": arms,
        "claim_boundary": (
            "Latency is measured for the deterministic layer only, with no LLM "
            "call in either arm. Quality is measured on the control decision "
            "the layer produces, not on generated prose: the layer-off arm has "
            "no decision to score, so its quality baseline is the empty "
            "default. These numbers show the layer's compute cost and the "
            "structural properties of its decision; they do not measure "
            "end-to-end response quality, which requires a paired human or "
            "model-level evaluation (see eval/AB_PROTOCOL.md)."
        ),
    }


def check_invariants(report: dict) -> list[str]:
    """Assertions that must hold for the numbers to mean anything."""
    failures: list[str] = []
    design = report["design"]
    expected_n = design["n_measurements_per_arm"]

    for name, arm in report["arms"].items():
        lat = arm["latency_ms"]
        if lat["layer_on"]["n"] != expected_n:
            failures.append(f"{name}: layer_on measured {lat['layer_on']['n']} times")
        if lat["layer_off"]["n"] != expected_n:
            failures.append(f"{name}: layer_off measured {lat['layer_off']['n']} times")
        # Classification + routing must not exceed the measured total.
        on = lat["layer_on"]
        parts = lat["classification_on"]["mean_ms"] + lat["routing_on"]["mean_ms"]
        if parts > on["mean_ms"]:
            failures.append(
                f"{name}: classification+routing ({parts:.4f} ms) exceeds total "
                f"({on['mean_ms']:.4f} ms)"
            )
        # The two classification halves must sum to the whole span.
        halves = (
            lat["classification_intent_on"]["mean_ms"]
            + lat["classification_affect_on"]["mean_ms"]
        )
        if abs(halves - lat["classification_on"]["mean_ms"]) > 0.01:
            failures.append(
                f"{name}: intent+affect ({halves:.4f} ms) does not sum to "
                f"classification ({lat['classification_on']['mean_ms']:.4f} ms)"
            )
        # p95 can never be below the median.
        for arm_key in ("layer_on", "layer_off", "added"):
            stats = lat[arm_key]
            if stats["p95_ms"] < stats["median_ms"] - 1e-9:
                failures.append(f"{name}/{arm_key}: p95 below median")
        # A layer with no decision can never match a labelled intent.
        if arm["quality"]["intent_label_accuracy"]["layer_off"] not in (None, 0.0):
            failures.append(
                f"{name}: layer_off matched a labelled intent despite having no "
                "intent decision"
            )
        # The layer-on arm must produce a decision on every case, so its
        # appropriateness baseline cannot be the empty default's.
        if arm["quality"]["appropriateness_rubric"]["layer_on"] <= 0.5:
            failures.append(
                f"{name}: layer_on appropriateness "
                f"{arm['quality']['appropriateness_rubric']['layer_on']} is too "
                "low to be a real decision"
            )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EQ layer A/B latency + quality benchmark"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    holdout = load_jsonl(args.holdout) if args.holdout.exists() else []

    report = run_benchmark(cases, holdout, runs=args.runs)

    failures = check_invariants(report)
    report["invariants"] = {"passed": not failures, "failures": failures}

    text = json.dumps(report, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)
    print(f"\nWrote {args.output}")

    if failures:
        for failure in failures:
            print(f"INVARIANT FAILURE: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
