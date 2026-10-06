from __future__ import annotations

import hashlib
import json
import random
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .auditor import audit_from_generation_metadata, regeneration_instruction
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


def select_stratified_cases(cases: list[dict], limit: int) -> list[dict]:
    """Select a deterministic, approximately balanced pilot across source strata.

    Strata are read from case["source"]["stratum"]. Selection preserves the
    within-stratum source order and interleaves strata round-robin so a simple
    prefix cannot accidentally become single-domain.
    """
    if limit < 1:
        raise ValueError("Stratified limit must be >= 1.")

    groups: dict[str, list[dict]] = {}
    for case in cases:
        source = case.get("source")
        stratum = source.get("stratum") if isinstance(source, dict) else None
        if not stratum:
            raise ValueError(
                "Stratified selection requires source.stratum on every case."
            )
        groups.setdefault(str(stratum), []).append(case)

    if not groups:
        raise ValueError("No strata available for stratified selection.")

    strata = sorted(groups)
    target = min(limit, len(cases))
    base = target // len(strata)
    remainder = target % len(strata)
    quotas = {
        stratum: base + (1 if index < remainder else 0)
        for index, stratum in enumerate(strata)
    }

    # If a stratum is too small, redistribute its unused quota deterministically.
    deficit = 0
    for stratum in strata:
        available = len(groups[stratum])
        if quotas[stratum] > available:
            deficit += quotas[stratum] - available
            quotas[stratum] = available

    while deficit > 0:
        progressed = False
        for stratum in strata:
            if quotas[stratum] < len(groups[stratum]):
                quotas[stratum] += 1
                deficit -= 1
                progressed = True
                if deficit == 0:
                    break
        if not progressed:
            break

    selected: list[dict] = []
    max_quota = max(quotas.values(), default=0)
    for index in range(max_quota):
        for stratum in strata:
            if index < quotas[stratum]:
                selected.append(groups[stratum][index])

    return selected[:target]


def stratum_counts(cases: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        source = case.get("source")
        stratum = source.get("stratum") if isinstance(source, dict) else None
        key = str(stratum) if stratum else "unstratified"
        counts[key] = counts.get(key, 0) + 1
    return counts

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
        raw_intent = self.intent.infer(messages)
        intent = self.intent.reconcile_with_dialogue_evidence(
            raw_intent,
            messages,
            tracked.subtext.signals,
        )
        selection = self.selector.select(tracked.state, intent)
        steer = Steer.build(
            selection.policy,
            intent,
            repair=tracked.repair,
            interaction_quality=tracked.interaction_quality,
            reference=tracked.reference,
        )
        instruction = steer.system_instruction()

        steered = [{"role": "system", "content": instruction}, *messages]
        metadata = {
            "policy": selection.policy.name,
            "factored_action": (
                steer.action.to_dict()
                if steer.action is not None
                else None
            ),
            "register": selection.policy.register.value,
            "intent_kind": intent.kind,
            "intent_confidence": intent.confidence,
            "intent_kind_raw": raw_intent.kind,
            "intent_belief_raw": (
                raw_intent.belief.as_dict()
                if raw_intent.belief is not None
                else None
            ),
            "intent_risk_decision_raw": (
                {
                    "action": raw_intent.risk_decision.action,
                    "selected_intent": raw_intent.risk_decision.selected_intent,
                    "expected_loss": raw_intent.risk_decision.expected_loss,
                    "losses": raw_intent.risk_decision.losses_dict(),
                    "rationale": raw_intent.risk_decision.rationale,
                }
                if raw_intent.risk_decision is not None
                else None
            ),
            "intent_belief": (
                intent.belief.as_dict()
                if intent.belief is not None
                else None
            ),
            "intent_belief_entropy": (
                intent.belief.normalized_entropy
                if intent.belief is not None
                else None
            ),
            "intent_risk_decision": (
                {
                    "action": intent.risk_decision.action,
                    "selected_intent": intent.risk_decision.selected_intent,
                    "expected_loss": intent.risk_decision.expected_loss,
                    "losses": intent.risk_decision.losses_dict(),
                    "rationale": intent.risk_decision.rationale,
                }
                if intent.risk_decision is not None
                else None
            ),
            "subtext": tracked.subtext.label,
            "subtext_evidence_level": tracked.subtext.evidence_level,
            "stance": tracked.stance.label,
            "stance_evidence_level": tracked.stance.evidence_level,
            "repair": {
                "active": tracked.repair.active,
                "kind": tracked.repair.kind,
                "target_turn_index": tracked.repair.target_turn_index,
                "target_excerpt": tracked.repair.target_excerpt,
                "correction_excerpt": tracked.repair.correction_excerpt,
                "replacement_excerpt": tracked.repair.replacement_excerpt,
                "replacement_explicit": tracked.repair.replacement_explicit,
                "repeated": tracked.repair.repeated,
                "recent_repair_count": tracked.repair.recent_repair_count,
                "confidence": tracked.repair.confidence,
                "evidence_level": tracked.repair.evidence_level,
                "evidence": list(tracked.repair.evidence),
            },
            "interaction_quality": {
                "current": tracked.interaction_quality.current,
                "delta": tracked.interaction_quality.delta,
                "repeated_failure_count": tracked.interaction_quality.repeated_failure_count,
                "unresolved_repair_count": tracked.interaction_quality.unresolved_repair_count,
                "clarification_count": tracked.interaction_quality.clarification_count,
                "evidence_level": tracked.interaction_quality.evidence_level,
                "evidence": list(tracked.interaction_quality.evidence),
            },
            "dialogue_reference": {
                "active": tracked.reference.active,
                "form": tracked.reference.form,
                "resolved": tracked.reference.resolved,
                "target_turn_index": tracked.reference.target_turn_index,
                "target_role": tracked.reference.target_role,
                "target_excerpt": tracked.reference.target_excerpt,
                "anchor_turn_index": tracked.reference.anchor_turn_index,
                "anchor_excerpt": tracked.reference.anchor_excerpt,
                "requires_clarification": tracked.reference.requires_clarification,
                "confidence": tracked.reference.confidence,
                "evidence_level": tracked.reference.evidence_level,
                "evidence": list(tracked.reference.evidence),
                "candidate_stack": [
                    {
                        "turn_index": candidate.turn_index,
                        "role": candidate.role,
                        "kind": candidate.kind,
                        "excerpt": candidate.excerpt,
                        "evidence": list(candidate.evidence),
                    }
                    for candidate in tracked.reference.candidate_stack
                ],
            },
            "selection_rationale": selection.rationale,
        }
        return steered, metadata


# Surface-form limits expressed without any policy content. Used to build the
# length-matched control arm: the model is asked for the same amount of text it
# is asked for in the EQ arm, but receives no affect, intent, task or repair
# decision. Without it, EQ-vs-baseline measures length as much as steering.
#
# The profiles exist because a single generic brevity instruction undershoots:
# on a 12-case Qwen pilot the first version left EQ at 0.793 of the control,
# still a 1.28x gap. A policy-derived instruction constrains harder than a
# generic one, so the control needs a dial.
#
# Calibration selects among these on length alone. Tuning the control against
# preference outcomes would fit the nuisance variable to the result it is meant
# to be compared against.
LENGTH_CONTROL_PROFILES: dict[str, str] = {
    "generic": (
        "Answer directly in as few sentences as the request needs. Use at most "
        "{question_budget} question(s), and do not add preamble or offers of further help."
    ),
    "terse": (
        "Reply in one or two sentences. Use at most {question_budget} question(s). "
        "No preamble, no restatement of the conversation, no offer of further help, "
        "no summary of what you are about to say."
    ),
    "minimal": (
        "Give the single most useful reply and nothing else. One or two sentences. "
        "Use at most {question_budget} question(s). Do not restate the conversation, "
        "do not add preamble, do not offer further help, do not summarise, and do "
        "not explain your reasoning."
    ),
    "one_line": (
        "Answer in one sentence if that is enough, otherwise two. Use at most "
        "{question_budget} question(s). No preamble, no restatement, no caveats, no "
        "offers of further help, no summary. If the request needs nothing more than "
        "a direct reply, give exactly that."
    ),
}

DEFAULT_LENGTH_PROFILE = "generic"


def build_length_matched_messages(
    messages: list[dict],
    realization: dict,
    profile: str = DEFAULT_LENGTH_PROFILE,
) -> list[dict]:
    """Baseline arm with only the surface budget pinned, not the EQ policy."""
    if profile not in LENGTH_CONTROL_PROFILES:
        raise ValueError(
            f"Unknown length profile {profile!r}; expected one of "
            f"{sorted(LENGTH_CONTROL_PROFILES)}"
        )
    budget = realization.get("question_budget", 0)
    instruction = LENGTH_CONTROL_PROFILES[profile].format(question_budget=budget)
    if realization.get("verbosity") != "low" and profile == DEFAULT_LENGTH_PROFILE:
        instruction = (
            "Answer directly in as few sentences as the request needs. "
            "Use at most "
            f"{budget} question(s), and do not add preamble or offers of further help."
        )
    return [{"role": "system", "content": instruction}, *messages]


def generate_pairs(
    cases: Iterable[dict],
    *,
    model: CommandModel,
    pipeline: FullEQPipeline,
    seed: int = 0,
    allow_annotations: bool = False,
    auditor_regenerations: int = 0,
    length_matched_arm: bool = False,
    length_profiles: tuple[str, ...] | None = None,
) -> list[dict]:
    if auditor_regenerations not in {0, 1}:
        raise ValueError("auditor_regenerations must be 0 or 1.")
    profiles = tuple(length_profiles or ())
    for name in profiles:
        if name not in LENGTH_CONTROL_PROFILES:
            raise ValueError(
                f"Unknown length profile {name!r}; expected one of "
                f"{sorted(LENGTH_CONTROL_PROFILES)}"
            )

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
            realization = (eq_meta.get("factored_action") or {}).get("realization") or {}
            # One arm per profile so a single generation pass covers the whole
            # calibration sweep. Each arm is an independent condition with the
            # same model, temperature and per-case seed.
            conditions = ["baseline", "eq"] + [
                f"length_matched:{name}" for name in profiles
            ]
            if length_matched_arm and not profiles:
                conditions.append("length_matched")
            rng.shuffle(conditions)
            responses: dict[str, str] = {}

            for condition in conditions:
                if condition == "baseline":
                    condition_messages = messages
                elif condition == "eq":
                    condition_messages = eq_messages
                else:
                    profile = (
                        condition.split(":", 1)[1]
                        if ":" in condition
                        else DEFAULT_LENGTH_PROFILE
                    )
                    condition_messages = build_length_matched_messages(
                        messages,
                        realization,
                        profile,
                    )
                responses[condition] = model.generate(
                    condition_messages,
                    seed=pair_seed,
                    condition=condition,
                )

            initial_eq = responses["eq"]
            initial_audit = audit_from_generation_metadata(
                initial_eq,
                messages,
                eq_meta,
            )
            final_audit = initial_audit
            regeneration_count = 0

            if auditor_regenerations and initial_audit.should_regenerate:
                instruction = regeneration_instruction(initial_audit)
                if instruction:
                    retry_messages = [
                        eq_messages[0],
                        {"role": "system", "content": instruction},
                        *eq_messages[1:],
                    ]
                    responses["eq"] = model.generate(
                        retry_messages,
                        seed=pair_seed,
                        condition="eq_audit_retry",
                    )
                    regeneration_count = 1
                    final_audit = audit_from_generation_metadata(
                        responses["eq"],
                        messages,
                        eq_meta,
                    )

            record = {
                "id": case_id,
                "source": case.get("source"),
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
                    "auditor_regenerations_allowed": auditor_regenerations,
                    "audit_regeneration_count": regeneration_count,
                    "response_audit_initial": initial_audit.to_dict(),
                    "response_audit": final_audit.to_dict(),
                    **eq_meta,
                },
            }
            if regeneration_count:
                record["eq_initial"] = initial_eq
            for name in profiles:
                record[f"length_matched_{name}"] = responses[f"length_matched:{name}"]
            if length_matched_arm and not profiles:
                record["length_matched"] = responses["length_matched"]
            if profiles:
                record["length_profiles"] = list(profiles)

            output.append(record)
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
    auditor_regenerations: int = 0,
    length_matched_arm: bool = False,
    length_profiles: tuple[str, ...] = (),
) -> dict:
    profiles = list(length_profiles)
    # Reject unknown names here as well as at generation time. The manifest is the
    # record other people read, so an unresolvable profile must not be written
    # into it and left to fail later.
    unknown = [name for name in profiles if name not in LENGTH_CONTROL_PROFILES]
    if unknown:
        raise ValueError(
            f"Unknown length profile(s) {unknown}; expected one of "
            f"{sorted(LENGTH_CONTROL_PROFILES)}"
        )
    arms = (
        ["baseline", "eq"]
        + [f"length_matched:{name}" for name in profiles]
        + (["length_matched"] if length_matched_arm and not profiles else [])
    )
    return {
        "design": (
            "same-base-model length-calibration sweep"
            if profiles
            else (
                "same-base-model three-arm baseline-vs-length-matched-vs-eq-layer"
                if length_matched_arm
                else "same-base-model paired baseline-vs-eq-layer"
            )
        ),
        "model_id": model.model_id,
        "model_command_executable": model.command[0],
        "model_command_argv_sha256": hashlib.sha256(
            "\0".join(model.command).encode("utf-8")
        ).hexdigest(),
        "temperature": model.temperature,
        "global_seed": global_seed,
        "n_cases": n_cases,
        "annotations_used": allow_annotations,
        "response_auditor_mode": (
            "one-shot-regeneration"
            if auditor_regenerations
            else "passive-metadata-only"
        ),
        "auditor_regenerations_allowed": auditor_regenerations,
        "cases_path": str(cases_path),
        "cases_sha256": file_sha256(cases_path),
        "affect_model_sha256": file_sha256(affect_model),
        "subtext_model_sha256": file_sha256(subtext_model),
        "eq_layer_git_commit": git_commit,
        "pairing_rule": "same model id, same sampling temperature, same per-case seed",
        "arms": arms,
        "length_matched_arm": length_matched_arm,
        "length_profiles": profiles,
        "length_profile_selection": (
            "Chosen on arm length only. The control must not be tuned against "
            "preference outcomes, or the nuisance variable gets fitted to the "
            "result it exists to be compared against."
        ),
        "arm_difference": (
            "EQ arm prepends only the EQ-Layer system control instruction; "
            "the original transcript is preserved identically in both arms."
        ),
        "length_control": (
            "The length-matched arm pins only the surface budget (verbosity, "
            "question_budget) with no affect, intent, task or repair decision. "
            "EQ-vs-length_matched isolates the deterministic policy contribution; "
            "EQ-vs-baseline does not, because the EQ arm runs roughly 3x shorter "
            "than the unconstrained baseline."
        ),
        "claim_boundary": (
            "EQ-vs-baseline mixes the deterministic decision with the surface budget "
            "it implies. Only EQ-vs-length_matched measures the deterministic layer "
            "alone. Report both."
        ),
    }
