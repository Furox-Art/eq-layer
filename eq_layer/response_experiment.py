from __future__ import annotations

import hashlib
import json
import random
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .policies import Selector
from .steer import Steer
from .tracker import ConversationTracker
from .trained_affect import TrainedAffect
from .trained_intent import TrainedIntent
from .trained_subtext import TrainedSubtext



LEAKAGE_FIELDS = {
    "expected_policy",
    "expected_intent",
    "expected_concepts",
    "annotated",
}


def validate_experiment_cases(cases: list[dict], *, final: bool = False) -> dict:
    errors: list[str] = []
    seen: set[str] = set()
    role_counts = {"user": 0, "assistant": 0, "system": 0, "other": 0}

    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id", "")).strip()
        if not case_id:
            errors.append(f"line {index}: missing id")
            continue
        if case_id in seen:
            errors.append(f"{case_id}: duplicate id")
        seen.add(case_id)

        messages = case.get("transcript") or case.get("context")
        if not isinstance(messages, list) or not messages:
            errors.append(f"{case_id}: missing transcript/context")
            continue
        if not any(
            isinstance(message, dict) and message.get("role") == "user"
            for message in messages
        ):
            errors.append(f"{case_id}: transcript contains no user turn")

        for message in messages:
            if not isinstance(message, dict):
                errors.append(f"{case_id}: non-object message")
                continue
            role = str(message.get("role", ""))
            role_counts[role if role in role_counts else "other"] += 1
            if not str(message.get("content", "")).strip():
                errors.append(f"{case_id}: empty message content")

        if final:
            leaked = sorted(LEAKAGE_FIELDS.intersection(case))
            if leaked:
                errors.append(
                    f"{case_id}: final held-out case contains development/gold fields: "
                    + ", ".join(leaked)
                )

    if final and len(cases) < 120:
        errors.append(
            f"final run has {len(cases)} cases; protocol target is at least 120 "
            "total cases to make ~90+ non-ties plausible"
        )

    return {
        "n_cases": len(cases),
        "unique_ids": len(seen),
        "role_counts": role_counts,
        "final_mode": final,
        "errors": errors,
        "ok": not errors,
    }

def stable_case_seed(global_seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CommandModel:
    """Provider-neutral model adapter.

    The configured command receives one JSON object on stdin and must write
    either plain response text or a JSON object containing `text`/`response`
    on stdout. The same command/model id is used for both experiment arms.
    """

    command: tuple[str, ...]
    model_id: str
    temperature: float = 0.0
    timeout_seconds: int = 120
    persistent: bool = False
    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_shell(
        cls,
        command: str,
        *,
        model_id: str,
        temperature: float = 0.0,
        timeout_seconds: int = 120,
        persistent: bool = False,
    ) -> "CommandModel":
        parsed = tuple(shlex.split(command))
        if not parsed:
            raise ValueError("Model command cannot be empty.")
        return cls(
            command=parsed,
            model_id=model_id,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            persistent=persistent,
        )

    def generate(
        self,
        messages: list[dict],
        *,
        seed: int,
        condition: str,
    ) -> str:
        payload = {
            "model_id": self.model_id,
            "seed": seed,
            "temperature": self.temperature,
            "messages": messages,
        }
        if self.persistent:
            raw = self._persistent_generate(payload)
        else:
            completed = subprocess.run(
                list(self.command),
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Model command failed ({completed.returncode}): "
                    f"{completed.stderr.strip()}"
                )
            raw = completed.stdout.strip()

        return self._decode_response(raw)

    def _persistent_generate(self, payload: dict) -> str:
        if self._process is None:
            self._process = subprocess.Popen(
                list(self.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

        process = self._process
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Persistent model command pipes are unavailable.")
        if process.poll() is not None:
            stderr = process.stderr.read().strip() if process.stderr else ""
            raise RuntimeError(
                f"Persistent model command exited ({process.returncode}): {stderr}"
            )

        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()
        raw = process.stdout.readline().strip()
        if not raw:
            stderr = process.stderr.read().strip() if process.stderr else ""
            raise RuntimeError(
                "Persistent model command returned an empty response. "
                f"stderr={stderr}"
            )
        return raw

    @staticmethod
    def _decode_response(raw: str) -> str:
        if not raw:
            raise RuntimeError("Model command returned an empty response.")

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return raw

        if isinstance(decoded, dict):
            for key in ("text", "response"):
                value = decoded.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        raise RuntimeError(
            "Model command JSON must contain a non-empty 'text' or 'response'."
        )

    def close(self) -> None:
        if self._process is None:
            return
        process = self._process
        self._process = None
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@dataclass
class FullEQPipeline:
    tracker: ConversationTracker
    intent: TrainedIntent
    selector: Selector

    @classmethod
    def load(
        cls,
        affect_model: str | Path,
        subtext_model: str | Path,
    ) -> "FullEQPipeline":
        return cls(
            tracker=ConversationTracker(
                affect=TrainedAffect.load(affect_model),
                subtext=TrainedSubtext.load(subtext_model),
            ),
            intent=TrainedIntent.from_bundled(),
            selector=Selector(),
        )

    def steer_messages(
        self,
        messages: list[dict],
        *,
        annotated: dict | None = None,
    ) -> tuple[list[dict], dict]:
        tracked = self.tracker.infer(
            messages,
            turn_index=len(messages),
            annotated=annotated or {},
        )
        intent = self.intent.infer(messages)
        selection = self.selector.select(tracked.state, intent)
        instruction = Steer.build(selection.policy, intent).system_instruction()

        steered = [{"role": "system", "content": instruction}, *messages]
        metadata = {
            "policy": selection.policy.name,
            "register": selection.policy.register.value,
            "intent_kind": intent.kind,
            "intent_confidence": intent.confidence,
            "subtext": tracked.subtext.label,
            "subtext_evidence_level": tracked.subtext.evidence_level,
            "stance": tracked.stance.label,
            "stance_evidence_level": tracked.stance.evidence_level,
            "selection_rationale": selection.rationale,
        }
        return steered, metadata


def generate_pairs(
    cases: Iterable[dict],
    *,
    model: CommandModel,
    pipeline: FullEQPipeline,
    seed: int = 0,
    allow_annotations: bool = False,
) -> list[dict]:
    rng = random.Random(seed)
    output: list[dict] = []
    seen: set[str] = set()

    try:
        for case in cases:
                case_id = str(case["id"])
            if case_id in seen:
                raise ValueError(f"Duplicate experiment case id: {case_id}")
            seen.add(case_id)

            messages = list(case.get("transcript") or case.get("context") or [])
            if not messages:
                raise ValueError(f"Experiment case has no transcript/context: {case_id}")

            annotated = case.get("annotated") if allow_annotations else None
            eq_messages, eq_meta = pipeline.steer_messages(
                messages,
                annotated=annotated,
            )

            pair_seed = stable_case_seed(seed, case_id)
            conditions = ["baseline", "eq"]
            rng.shuffle(conditions)
            responses: dict[str, str] = {}

            for condition in conditions:
                condition_messages = messages if condition == "baseline" else eq_messages
                responses[condition] = model.generate(
                    condition_messages,
                    seed=pair_seed,
                    condition=condition,
                )

            output.append(
                {
                    "id": case_id,
                    "context": messages,
                    "baseline": responses["baseline"],
                    "eq": responses["eq"],
                    "generation": {
                        "model_id": model.model_id,
                        "temperature": model.temperature,
                        "pair_seed": pair_seed,
                        "order": conditions,
                        "same_model_both_arms": True,
                        "annotations_used": bool(annotated),
                        **eq_meta,
                    },
                }
            )

    finally:
        model.close()
    return output


def experiment_manifest(
    *,
    model: CommandModel,
    affect_model: str | Path,
    subtext_model: str | Path,
    cases_path: str | Path,
    global_seed: int,
    n_cases: int,
    allow_annotations: bool,
    git_commit: str | None = None,
) -> dict:
    return {
        "design": "same-base-model paired baseline-vs-eq-layer",
        "model_id": model.model_id,
        "model_command_executable": model.command[0],
        "model_command_argv_sha256": hashlib.sha256(
            "\0".join(model.command).encode("utf-8")
        ).hexdigest(),
        "temperature": model.temperature,
        "global_seed": global_seed,
        "n_cases": n_cases,
        "annotations_used": allow_annotations,
        "cases_path": str(cases_path),
        "cases_sha256": file_sha256(cases_path),
        "affect_model_sha256": file_sha256(affect_model),
        "subtext_model_sha256": file_sha256(subtext_model),
        "eq_layer_git_commit": git_commit,
        "pairing_rule": "same model id, same sampling temperature, same per-case seed",
        "arm_difference": (
            "EQ arm prepends only the EQ-Layer system control instruction; "
            "the original transcript is preserved identically in both arms."
        ),
    }
