# Response-Level A/B Protocol

This protocol tests one claim only:

> Does adding EQ-Layer control improve the final response from the **same base model**?

It is separate from component benchmarks for intent, affect, dialogue act, or
subtext.

## Primary hypothesis

On held-out conversations, raters prefer the EQ-Layer response over the plain
baseline on **overall response quality** more often than chance among non-tied
cases.

Primary endpoint:

- case-level majority vote on `overall`
- EQ win rate among non-ties
- exact two-sided sign test against 0.5
- Wilson 95% confidence interval

Secondary endpoints:

- `intent_fidelity`
- `appropriateness`
- `actionability`
- `non_patronizing`

Secondary endpoints are descriptive unless a separate multiplicity plan is
registered before rating.

## Conditions

Both arms must use:

- the exact same base model/version
- the exact same decoding temperature
- the same per-case seed when the backend supports a seed
- the exact same original transcript
- the same external tools/data availability

The only intended difference is:

- **baseline:** original transcript
- **EQ:** the same transcript plus EQ-Layer's generated system control instruction

Do not add a different system persona, safety policy, retrieval result, tool
result, or hidden context to one arm only.

## Leakage controls

Final claims must not use `eval/cases.jsonl`. That file is a development set.

For a final run:

1. Freeze a held-out/external case file before generating either arm.
2. Record its SHA-256 in the experiment manifest.
3. Do not edit cases after seeing paired outputs.
4. Default to `annotations_used=false`.
5. Do not use expected policy, expected concept, stance, or subtext gold labels
   to steer the EQ arm.
6. Keep the A/B decode key hidden from raters.

Oracle-annotation runs are permitted only as explicitly labelled diagnostic
analyses and must not be reported as the production EQ-Layer result.

## Raters

Preferred design:

- at least 3 independent raters
- raters see only randomized A/B responses and the original conversation
- raters do not know which side is EQ-Layer
- one vote per dimension: `A`, `B`, or `tie`

The multi-rater analysis uses **one case as the statistical unit**. It first
takes the majority preference within each case, then runs the sign test across
cases. Individual rater votes are not incorrectly treated as independent
samples.

Inter-rater agreement is reported with Fleiss' kappa.

## Sample-size target

A practical minimum target is **90 non-tied cases**. Under a simple two-sided
exact sign-test design, that is around the scale needed for 80% power when the
true EQ preference rate is approximately 65% rather than 50%.

Because ties reduce the effective sample, recruit more than 90 total cases.
A target of 120-150 held-out cases is preferable when feasible.

This is a planning target, not a guarantee of power for every tie rate or
effect size.

## Generation

Example:

```bash
python eval/ab_generate.py heldout.jsonl pairs.jsonl manifest.json \
  --model-command "python model_adapter.py" \
  --model-id SAME_BASE_MODEL_VERSION \
  --affect-model artifacts/emobank_affect.joblib \
  --subtext-model artifacts/xdailydialog_subtext.joblib \
  --temperature 0 \
  --seed 42 \
  --git-commit <eq-layer-commit>
```

The model command receives JSON on stdin with:

- `model_id`
- `seed`
- `temperature`
- `messages`

The backend is intentionally **not** told whether a call belongs to the
baseline or EQ arm. The arm assignment remains inside the experiment engine,
so the backend cannot branch on a condition label.

It must write either plain response text or JSON containing `text` or
`response`.

Secrets should be passed through environment variables, not command-line
arguments. The manifest stores only the executable name and a SHA-256
fingerprint of command argv.

## Blinding

```bash
python eval/ab_prepare.py pairs.jsonl ballot.jsonl key.json --seed 42
```

Keep `key.json` hidden until all ratings are complete.

## Scoring

Single rater:

```bash
python eval/ab_score.py rated_ballot.jsonl key.json
```

Multiple raters:

```bash
python eval/ab_score_multi.py key.json rater1.jsonl rater2.jsonl rater3.jsonl
```

## Interpretation rule

A positive component benchmark is not evidence that EQ-Layer improves final
responses.

A final response-level claim should be made only after:

- held-out generation is frozen
- blinding is preserved
- the primary endpoint is scored
- the effect size and 95% CI are reported
- no-go / null / mixed results are retained
