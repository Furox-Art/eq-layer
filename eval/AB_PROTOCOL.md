# Evaluation Protocol

Two claims, deliberately not the same claim, and ranked by how well each can
actually be measured.

## Primary: the routing decision

> Does EQ-Layer pick the action a careful reader would pick, on conversations
> it has not seen?

Measured on the frozen external held-out set with no model backend in the
path. Each case gets one adjudicated task move and the layer's `task_move` is
compared against it.

- unit: case-level agreement between `FactoredAction.task_move` and the
  adjudicated move
- statistic: binomial proportion with a Wilson 95% interval
- headline strata: `empathetic` and `task_general` only

`task_repair` is excluded from the headline and reported separately. It is
selected into the held-out set by `REPAIR_PHRASES` and detected by
`CORRECTION_MARKERS`, which share six of seven phrases, so agreement there is
near-vacuous and would inflate a pooled number. Cases where
`intent_kind` is `unknown` are excluded from the headline too: they land on the
clarify/respond fallback, so pooling them hides how much agreement is a
guessing convention.

Scripts: `eval/route_heldout_eval.py`, `eval/route_score.py`.

This endpoint has no length problem. The decision is an action label, not
prose, so nothing about it can be an artefact of how much the model wrote.

## Secondary: the rendered response

> Does the steered model produce a response a careful reader prefers over the
> same model without steering?

This is the original claim and it is **not currently supportable**, for a
measured reason rather than a suspected one.

On the frozen 120-case set the EQ arm runs 15.1 mean words against a baseline
at 42.9. A calibrated length-matched control was built to close that gap:

- positive, count-based instructions moved length far more than negative ones;
  prohibitions made replies longer on a small model, mean characters 97 for a
  mild profile against 155 for the strictest
- the ladder `shortest_complete` / `one_sentence` / `two_sentence` /
  `three_sentence` bottoms out at **11.2 percent** off, outside the 10 percent
  tolerance
- `eq_matched`, which reads its sentence budget off the EQ response, made it
  worse at 58.3 percent, because the model overshoots the requested count and
  EQ's brevity comes from its decision rather than from a surface budget

The residual is structural: a surface instruction cannot reproduce the length a
decision-derived instruction produces. Anyone rerunning this must not read a
preference win as an EQ effect without stating the residual.

If this endpoint is pursued, the ballot compares EQ against the calibrated
control rather than the raw baseline, and `eval/prepare_selected_ballot.py`
names the control in the decode key so the artifact does not misdescribe it.
Selection stays on length only; tuning the control against preference would
fit the nuisance variable to the outcome it exists to be compared against.

Note that coupling is not free: reading a per-case target off the EQ response
pushes the two arms toward ties, which hides an EQ win rather than manufacturing
one.

### Statistics, if ratings are collected

- case-level majority vote on `overall`
- EQ win rate among non-ties
- exact two-sided sign test against 0.5
- Wilson 95% confidence interval

`intent_fidelity`, `appropriateness`, `actionability` and `non_patronizing` are
secondary and descriptive unless a multiplicity plan is registered before
rating.

Rows where both responses are identical carry no information and must be
counted separately rather than folded into ties. One such row appeared in the
120-case ballot (`external-036`).

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

The two endpoints need different raters and must not be pooled.

**Decision endpoint.** One adjudicator per case names the task move. This is
roughly 120 labels for the full held-out set, not 120 x 5 pairwise ratings, and
the label is closer to objective than a quality judgement. Independent
adjudicators are still preferred: two agreeing adjudicators, with a third for
disagreement, beats one rater whose consistency is never checked.

**Response endpoint.** The design below, if the endpoint is pursued at all.

- at least 3 independent raters
- raters see only randomized A/B responses and the original conversation
- raters do not know which side is EQ-Layer
- one vote per dimension: `A`, `B`, or `tie`

The multi-rater analysis uses **one case as the statistical unit**. It first
takes the majority preference within each case, then runs the sign test across
cases. Individual rater votes are not incorrectly treated as independent
samples.

Inter-rater agreement is reported with Fleiss' kappa.

## Reproducibility

The backend must be deterministic for the pairing rule to mean anything. At
temperature 0 with a fixed per-case seed, CPU matmul reduction order varies
between runs and flips the argmax at near-ties: two pilot runs of the same
commit produced identical text for only 8 of 12 EQ arms and 7 of 12 baseline
arms.

`eval/hf_model_server.py` therefore enables deterministic algorithms and
single-threaded execution by default. `eval/reproducibility_check.py` measures
it rather than asserting it, sending the same payload repeatedly inside one
process and again in a fresh one. A pilot that skips this check cannot claim a
paired design.

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

Routing agreement is not evidence that EQ-Layer improves final responses
either. It shows the decision matches an adjudicated move. Whether a better
decision produces a better reply is a separate question, and it is currently
the unsupported one, because the response endpoint cannot be length-matched
below an 11 percent residual.

A final decision-level claim should be made only after:

- the held-out set is frozen and its SHA-256 recorded
- the ballot is blinded and the stratum exclusions are applied
- agreement is reported with a Wilson interval
- circular strata and unknown-intent cases are reported separately, not dropped
  silently
- null and mixed results are retained

A final response-level claim additionally requires human ratings, and must
state the measured length residual rather than describing the arms as matched.

## Current state

Measured, reproducible, and not yet a result:

| item | value |
|---|---|
| held-out cases | 120, frozen, `d53fcaae…` |
| routing decision cases | 120 generated, 90 in the headline after exclusions |
| adjudicated labels | **0** |
| response ballot | 120 rows, blinded, ratings blank |
| length residual vs calibrated control | 11.2 percent, outside tolerance |
| generated responses | none rated |

No claim about EQ-Layer improving anything has been established. The next step
is human adjudication of `route_gold_ballot.jsonl`, roughly one label per case.
