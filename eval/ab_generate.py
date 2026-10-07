"""Generate same-base-model baseline vs EQ-Layer response pairs.

The model is supplied as a command-line adapter so this runner stays provider
neutral. The command receives JSON on stdin and returns response text on stdout.

Final scientific runs reject eval/cases.jsonl by default because it is a
development/seed set. Use a held-out or external case file instead.

Usage:
  python eval/ab_generate.py heldout.jsonl pairs.jsonl manifest.json \
    --model-command "python my_model_adapter.py" \
    --model-id my-model \
    --affect-model artifacts/emobank_affect.joblib \
    --subtext-model artifacts/xdailydialog_subtext.joblib \
    --git-commit <commit>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.ab_eval import load_jsonl, write_jsonl  # noqa: E402
from eq_layer.response_experiment import (  # noqa: E402
    EQ_MATCHED_PROFILE,
    LENGTH_CONTROL_PROFILES,
    CommandModel,
    FullEQPipeline,
    experiment_manifest,
    generate_pairs,
    select_stratified_cases,
    stratum_counts,
    validate_experiment_cases,
)


DEVELOPMENT_CASES = Path(__file__).with_name("cases.jsonl").resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases")
    parser.add_argument("pairs")
    parser.add_argument("manifest")
    parser.add_argument("--model-command", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--affect-model", required=True)
    parser.add_argument("--subtext-model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--auditor-regenerations",
        type=int,
        choices=(0, 1),
        default=0,
        help="Engineering mode: allow at most one EQ retry after a hard structural audit failure.",
    )
    parser.add_argument(
        "--length-matched-arm",
        action="store_true",
        help=(
            "Add a third arm that pins only the surface budget (verbosity, "
            "question_budget) and receives no EQ-Layer decision. EQ-vs-length_matched "
            "isolates the deterministic policy contribution from the length confound; "
            "EQ-vs-baseline does not."
        ),
    )
    parser.add_argument(
        "--length-profile",
        action="append",
        dest="length_profiles",
        choices=sorted(LENGTH_CONTROL_PROFILES) + [EQ_MATCHED_PROFILE],
        default=None,
        help=(
            "Generate one length-matched control arm per profile so a single pass "
            "covers the calibration sweep. Repeatable. Selection is made on arm "
            "length only, never on preference outcomes. eq_matched derives its "
            "sentence budget from the EQ response, which couples the arms toward "
            "ties."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--persistent-model-command",
        action="store_true",
        help="Keep one model-command process alive and exchange newline-delimited JSON.",
    )
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stratified-limit", type=int, default=None)
    parser.add_argument(
        "--final",
        action="store_true",
        help="Enforce final-run held-out/leakage/sample-size checks before generation.",
    )
    parser.add_argument(
        "--allow-development-cases",
        action="store_true",
        help="Pilot/debug only. Do not use for a final claim.",
    )
    parser.add_argument(
        "--allow-oracle-annotations",
        action="store_true",
        help="Oracle analysis only. Default final runs ignore case annotations.",
    )
    args = parser.parse_args()

    if args.final and args.limit is not None:
        parser.error("--final cannot be combined with --limit")
    if args.final and args.stratified_limit is not None:
        parser.error("--final cannot be combined with --stratified-limit")
    if args.limit is not None and args.stratified_limit is not None:
        parser.error("--limit and --stratified-limit are mutually exclusive")
    if args.final and args.allow_development_cases:
        parser.error("--final cannot use --allow-development-cases")
    if args.final and args.allow_oracle_annotations:
        parser.error("--final cannot use --allow-oracle-annotations")
    if args.final and args.auditor_regenerations:
        parser.error(
            "--final cannot use auditor regeneration until that protocol is separately preregistered"
        )

    cases_path = Path(args.cases).resolve()
    if cases_path == DEVELOPMENT_CASES and not args.allow_development_cases:
        parser.error(
            "eval/cases.jsonl is a development set. "
            "Use held-out/external cases or pass --allow-development-cases for pilot debugging."
        )

    cases = load_jsonl(cases_path)
    preflight = validate_experiment_cases(cases, final=args.final)
    if not preflight["ok"]:
        parser.error("A/B case preflight failed: " + " | ".join(preflight["errors"]))

    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be >= 1")
        cases = cases[: args.limit]
    elif args.stratified_limit is not None:
        if args.stratified_limit < 1:
            parser.error("--stratified-limit must be >= 1")
        try:
            cases = select_stratified_cases(cases, args.stratified_limit)
        except ValueError as exc:
            parser.error(str(exc))

    model = CommandModel.from_shell(
        args.model_command,
        model_id=args.model_id,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        persistent=args.persistent_model_command,
    )
    pipeline = FullEQPipeline.load(
        affect_model=args.affect_model,
        subtext_model=args.subtext_model,
    )

    pairs = generate_pairs(
        cases,
        model=model,
        pipeline=pipeline,
        seed=args.seed,
        allow_annotations=args.allow_oracle_annotations,
        auditor_regenerations=args.auditor_regenerations,
        length_matched_arm=args.length_matched_arm,
        length_profiles=tuple(args.length_profiles or ()),
    )
    write_jsonl(args.pairs, pairs)

    manifest = experiment_manifest(
        model=model,
        affect_model=args.affect_model,
        subtext_model=args.subtext_model,
        cases_path=cases_path,
        global_seed=args.seed,
        n_cases=len(cases),
        allow_annotations=args.allow_oracle_annotations,
        git_commit=args.git_commit,
        auditor_regenerations=args.auditor_regenerations,
        length_matched_arm=args.length_matched_arm,
        length_profiles=tuple(args.length_profiles or ()),
    )
    manifest["development_cases_allowed"] = args.allow_development_cases
    manifest["final_mode"] = args.final
    manifest["case_preflight"] = preflight
    manifest["limit"] = args.limit
    manifest["stratified_limit"] = args.stratified_limit
    manifest["selected_strata"] = stratum_counts(cases)
    manifest["persistent_model_command"] = args.persistent_model_command
    manifest["auditor_regenerations"] = args.auditor_regenerations
    manifest["length_matched_arm"] = args.length_matched_arm

    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"generated {len(pairs)} paired responses")
    print(f"pairs={args.pairs}")
    print(f"manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
