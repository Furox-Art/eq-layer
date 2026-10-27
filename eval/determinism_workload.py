"""Deterministic control-layer workload sampled by the determinism matrix."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


def main() -> None:
    # The payload contains combining marks that a legacy console code page
    # cannot encode; pin stdout so standalone runs emit the same UTF-8 bytes as
    # the matrix, which already sets PYTHONIOENCODING=utf-8 for its variants.
    sys.stdout.reconfigure(encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from eq_layer.actions import compose_action
    from eq_layer.affect import HeuristicAffect
    from eq_layer.auditor import AuditContext, ResponseAuditor
    from eq_layer.intent_belief import (
        IntentBelief,
        decide_intent_action,
        fuse_dialogue_evidence,
    )
    from eq_layer.policies import Selector
    from eq_layer.response_experiment import select_stratified_cases
    from eq_layer.textnorm import normalise_light, normalise_match, normalise_plain

    messages = [
        {"role": "user", "content": "Neden bu kadar yavaş?"},
        {"role": "assistant", "content": "Bakıyorum"},
        {"role": "user", "content": "Neden yine?!"},
        {"role": "assistant", "content": "Kontrol ediyorum"},
        {"role": "user", "content": "Bu sefer gerçekten çözelim."},
    ]
    state = HeuristicAffect().infer(messages, turn_index=4)
    belief = IntentBelief.from_mapping(
        {"question": 3.0, "explanation": 2.0, "statement": 1.0, "action_request": 1.5}
    )
    fused, evidence = fuse_dialogue_evidence(
        belief,
        text="Neden yine?",
        dialogue_act="question",
        act_confidence=0.7,
    )
    decision = decide_intent_action(fused)
    selection = Selector().select(state)
    action = compose_action(selection.policy)
    audit = ResponseAuditor().audit(
        "Haklısınız, bu kez kontrol ediyorum ve sonucu paylaşacağım.",
        AuditContext(
            task_move=action.task_move,
            social_move=action.social_move,
            repair_move=action.repair_move,
            verbosity=action.realization.verbosity,
            question_budget=1,
            no_guess=True,
        ),
    )
    # The pilot selector is documented as deterministic: strata are read in
    # sorted order and the round-robin interleave is what keeps a prefix from
    # becoming single-domain. Source order is deliberately not sorted, so a
    # regression to set iteration changes the selection with the hash seed.
    pilot_cases = [
        {"id": f"{stratum}-{index}", "source": {"stratum": stratum}}
        for stratum, count in (("gamma", 4), ("alpha", 1), ("beta", 3))
        for index in range(count)
    ]
    pilot = select_stratified_cases(pilot_cases, limit=6)
    payload: dict[str, Any] = {
        "affect": asdict(state),
        "belief": fused.as_dict(),
        "evidence": list(evidence),
        "decision": {
            "action": decision.action,
            "selected_intent": decision.selected_intent,
            "expected_loss": decision.expected_loss,
            "losses": list(decision.losses),
            "rationale": decision.rationale,
        },
        "selection": {
            "policy": selection.policy.name,
            "rationale": selection.rationale,
            "runner_up": list(selection.runner_up),
        },
        "action": action.to_dict(),
        "audit": audit.to_dict(),
        "normalised": [
            normalise_light("  Neden  YİNE?!  "),
            normalise_plain("Neden yine?"),
            normalise_match("Neden yine?"),
        ],
        "pilot": [case["id"] for case in pilot],
    }
    if not audit.passed or selection.policy.name == "":
        raise SystemExit("workload produced an unexpected control decision; it is vacuous")
    if len(pilot) != 6 or len({case["id"].split("-", 1)[0] for case in pilot}) != 3:
        raise SystemExit("pilot selection is empty or incomplete; it is vacuous")
    artifact_dir = os.environ.get("DETERMINISM_ARTIFACT_DIR")
    if artifact_dir:
        Path(artifact_dir, "payload.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
