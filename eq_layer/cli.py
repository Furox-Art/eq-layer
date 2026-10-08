"""Command-line interface for the EQ-Layer control pass."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict


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
    from .intent import HeuristicIntent
    from .trained_intent import TrainedIntent

    if mode == "heuristic":
        return HeuristicIntent(), "heuristic"
    try:
        return TrainedIntent.from_bundled(), "trained-tfidf-logreg"
    except ImportError:
        if mode == "learned":
            raise
        return HeuristicIntent(), "heuristic-fallback-no-ml-extra"


def _lightweight_route(messages: list[dict], intent_mode: str) -> dict:
    from .affect import HeuristicAffect
    from .dialogue_reference import infer_dialogue_reference
    from .interaction import infer_interaction_quality, infer_repair_state
    from .policies import Selector
    from .steer import Steer

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


def _full_route(messages: list[dict], affect_model: str, subtext_model: str) -> dict:
    from .response_experiment import FullEQPipeline

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


def _route(args: argparse.Namespace) -> int:
    raw = args.messages_json if args.messages_json is not None else sys.stdin.read()
    if not raw.strip():
        raise ValueError("Provide transcript JSON with --messages-json or stdin")
    messages = _load_messages(raw)
    if bool(args.affect_model) != bool(args.subtext_model):
        raise ValueError("--affect-model and --subtext-model must be supplied together")
    if args.affect_model:
        result = _full_route(messages, args.affect_model, args.subtext_model)
    else:
        result = _lightweight_route(messages, args.intent_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


def _doctor() -> int:
    payload = {
        "package": "eq-layer",
        "version": "0.2.0",
        "python": sys.version.split()[0],
        "ml_extra_available": True,
    }
    try:
        import sklearn  # noqa: F401
    except ImportError:
        payload["ml_extra_available"] = False
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eq-layer")
    sub = parser.add_subparsers(dest="command", required=True)

    route = sub.add_parser("route", help="Emit an inspectable EQ-Layer control decision.")
    route.add_argument("--messages-json")
    route.add_argument(
        "--intent-mode",
        choices=("auto", "learned", "heuristic"),
        default="auto",
    )
    route.add_argument("--affect-model")
    route.add_argument("--subtext-model")
    route.add_argument("--pretty", action="store_true")
    route.set_defaults(handler=_route)

    doctor = sub.add_parser("doctor", help="Report local EQ-Layer runtime availability.")
    doctor.set_defaults(handler=lambda _args: _doctor())
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, json.JSONDecodeError, ImportError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
