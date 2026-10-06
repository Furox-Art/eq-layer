#!/usr/bin/env python3
"""Emit an inspectable EQ-Layer control decision for a visible transcript."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def _make_repo_importable() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "eq_layer").is_dir():
            sys.path.insert(0, str(parent))
            return


_make_repo_importable()

from eq_layer.affect import HeuristicAffect  # noqa: E402
from eq_layer.dialogue_reference import infer_dialogue_reference  # noqa: E402
from eq_layer.intent import HeuristicIntent  # noqa: E402
from eq_layer.interaction import infer_interaction_quality, infer_repair_state  # noqa: E402
from eq_layer.policies import Selector  # noqa: E402
from eq_layer.steer import Steer  # noqa: E402
from eq_layer.trained_intent import TrainedIntent  # noqa: E402


def _load_messages(raw: str) -> list[dict]:
    payload = json.loads(raw)
    if isinstance(payload, dict):
        payload = payload.get("messages")
    if not isinstance(payload, list) or not payload:
        raise ValueError("Input must be a non-empty JSON list or {messages:[...]}.")
    messages: list[dict] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"message {index} must be an object")
        role = str(item.get("role") or "")
        content = str(item.get("content") or "")
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"message {index} has unsupported role {role!r}")
        if not content.strip():
            raise ValueError(f"message {index} has empty content")
        messages.append({"role": role, "content": content})
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("Transcript must contain at least one user turn.")
    return messages


def _intent_adapter(mode: str):
    if mode == "heuristic":
        return HeuristicIntent(), "heuristic"
    try:
        return TrainedIntent.from_bundled(), "trained-tfidf-logreg"
    except ImportError:
        if mode == "learned":
            raise
        return HeuristicIntent(), "heuristic-fallback-no-ml-extra"


def lightweight_route(messages: list[dict], intent_mode: str) -> dict:
    state = HeuristicAffect().infer(messages, turn_index=len(messages))
    intent_adapter, intent_backend = _intent_adapter(intent_mode)
    intent = intent_adapter.infer(messages)
    repair = infer_repair_state(messages)
    interaction_quality = infer_interaction_quality(messages)
    reference = infer_dialogue_reference(messages)
    selection = Selector().select(state, intent)
    steer = Steer.build(
        selection.policy,
        intent,
        repair=repair,
        interaction_quality=interaction_quality,
        reference=reference,
    )
    return {
        "runtime_mode": "lightweight-local",
        "intent_backend": intent_backend,
        "policy": {
            "name": selection.policy.name,
            "register": selection.policy.register.value,
        },
        "affect": asdict(state),
        "intent": asdict(intent),
        "repair": asdict(repair),
        "interaction_quality": asdict(interaction_quality),
        "reference": asdict(reference),
        "factored_action": steer.action.to_dict() if steer.action else None,
        "system_instruction": steer.system_instruction(),
        "claim_boundary": (
            "Lightweight routing is a portable control pass, not the full learned "
            "research pipeline and not proof of emotional intelligence."
        ),
    }


def full_route(messages: list[dict], affect_model: str, subtext_model: str) -> dict:
    from eq_layer.response_experiment import FullEQPipeline

    pipeline = FullEQPipeline.load(affect_model, subtext_model)
    steered, metadata = pipeline.steer_messages(messages)
    instruction = (
        steered[0]["content"]
        if steered and steered[0].get("role") == "system"
        else ""
    )
    return {
        "runtime_mode": "full-learned",
        "intent_backend": "trained-tfidf-logreg",
        "metadata": metadata,
        "factored_action": metadata.get("factored_action"),
        "system_instruction": instruction,
        "claim_boundary": (
            "Full learned routing uses trained EQ-Layer components; response-quality "
            "improvement still requires empirical evaluation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages-json")
    parser.add_argument(
        "--intent-mode",
        choices=("auto", "learned", "heuristic"),
        default="auto",
    )
    parser.add_argument("--affect-model")
    parser.add_argument("--subtext-model")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    if bool(args.affect_model) != bool(args.subtext_model):
        parser.error("--affect-model and --subtext-model must be supplied together")

    raw = args.messages_json if args.messages_json is not None else sys.stdin.read()
    if not raw.strip():
        parser.error("Provide transcript JSON with --messages-json or stdin")

    try:
        messages = _load_messages(raw)
        if args.affect_model:
            result = full_route(messages, args.affect_model, args.subtext_model)
        else:
            result = lightweight_route(messages, args.intent_mode)
    except (ValueError, json.JSONDecodeError, ImportError) as exc:
        parser.error(str(exc))

    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
