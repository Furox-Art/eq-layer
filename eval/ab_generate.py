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
    CommandModel,
    FullEQPipeline,
    experiment_manifest,
    generate_pairs,
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--limit", type=int, default=None)
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

    model = CommandModel.from_shell(
        args.model_command,
        model_id=args.model_id,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
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
    )
    manifest["development_cases_allowed"] = args.allow_development_cases
    manifest["final_mode"] = args.final
    manifest["case_preflight"] = preflight
    manifest["limit"] = args.limit

    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"generated {len(pairs)} paired responses")
    print(f"pairs={args.pairs}")
    print(f"manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
