# eq-layer

A control layer that gives an LLM a *decision* about how to speak before it speaks — instead of asking it to be more empathetic in a prompt.

## Why this exists

Asking a model to be more empathetic is style transfer. It produces *more* of the same flat warmth, which reads as uncanny rather than attentive. The failure modes are structural, not knowledge gaps:

- **Flat affect.** "That sounds really tough!" arrives at the same register regardless of what was said.
- **Escalation blindness.** The user is frustrated, then angry, then threatening. The model holds one tone for all three turns.
- **Validation without motion.** The feeling is acknowledged; the next step never comes.
- **Repair collapse.** After a correction the model either capitulates entirely or starts defending itself. There is no middle.

So this repo does not retrain the model. It adds a layer:

1. **Affect state** — `{valence, arousal, escalation_delta, stance, subtext}` as an explicit intermediate representation, tracked across turns. `escalation_delta` is the point: snapshot classification cannot see a trend.
2. **Intent state** — `{kind, canonical_request, response_mode, confidence, constraints}` rewrites the request into an inspectable control signal without inventing missing details.
3. **Policy selection** — affect and intent jointly choose a response *move* (`mirror`, `direct`, `execute`, `repair`, `hold`, `boundary`, ...). Selection is a discrete decision, not a generation.
4. **Steering** — the chosen policy plus the canonical request are injected into the decode path so the decision actually lands in the tokens.

## Status

Early. The harness runs; the numbers are development signals, not a real-world EQ benchmark.

There is no standard benchmark for EQ. `eval/` is a first attempt at one, not a finished measurement. If you improve the scorer, that contribution matters as much as adding a policy.

## Quick start

Install the learned intent tracker:

```bash
pip install "eq-layer[ml]"
```

```python
from eq_layer import ConversationTracker, Selector, Steer, TrainedAffect, TrainedIntent, TrainedSubtext

messages = [
    {"role": "user", "content": "Randevumu üç kez değiştirdiler"},
]
affect = TrainedAffect.load("artifacts/emobank_affect.joblib")
subtext = TrainedSubtext.load("artifacts/xdailydialog_subtext.joblib")
tracker = ConversationTracker(affect=affect, subtext=subtext)
tracked = tracker.infer(messages)
state = tracked.state
intent = TrainedIntent.from_bundled().infer(messages)
selection = Selector().select(state, intent)
print(selection.policy.name, intent.canonical_request)

prompt = Steer.build(selection.policy, intent).apply_to_prompt(messages[-1]["content"])
```

Run the seeded cases. Exits non-zero on any mismatch or any policy that no
case can reach:

```bash
python eval/run.py
```

Run the intent benchmark separately:

```bash
python eval/intent_eval.py
```

It compares the learned adapter with the heuristic fallback and reports
five-fold cross-validation on the bundled corpus.

Train the affect regressor from the pinned official EmoBank release:

```bash
python tools/train_emobank_affect.py --output artifacts/emobank_affect.joblib
python eval/emobank_affect_eval.py
```

EmoBank is not bundled into this repository. The training/evaluation path uses
the upstream train/dev/test split and records its pinned commit and
CC-BY-SA-4.0 provenance in `THIRD_PARTY_DATA.md`.

Train the learned dialogue-signal tracker from pinned XDailyDialog files:

```bash
python tools/train_xdailydialog_subtext.py --output artifacts/xdailydialog_subtext.joblib
python eval/xdailydialog_subtext_eval.py
```

XDailyDialog provides supervised dialogue-act and basic-emotion labels. EQ-Layer
does **not** pretend that `challenge`, `demand`, or `exhaustion` are direct
upstream labels: those are conservative compositions of learned dialogue
signals with learned V/A state. See `THIRD_PARTY_DATA.md` for the license-chain
caution on the English data.

## Intent is a control signal, not mind-reading

The default intent path is now learned rather than keyword-selected.
`TrainedIntent` fits a balanced logistic-regression classifier over combined
word and character TF-IDF features. The bundled corpus currently contains
Turkish and English examples for `action_request`, `status_check`,
`explanation`, `question`, `statement`, and `unknown`.

The classifier emits a probability. Predictions below the confidence threshold
become `unknown` and request clarification instead of being silently promoted
to a guessed task. Explicit constraints such as `brief`, `scope_limited`,
and `no_guess` remain deterministic because they are control requirements,
not semantic labels.

`HeuristicIntent` remains available as a zero-dependency fallback. It is no
longer the adapter used by the main evaluation harness.

Affect can still override intent during sustained escalation. A clear action
request should drive the response in a calm turn; it should not erase a
multi-turn escalation signal.

The bundled data is small and synthetic. `eval/intent_eval.py` therefore
reports both a fixed held-out score and five-fold cross-validation; neither is
presented as evidence of production-level semantic understanding.

## Verified development baselines

The current CI-verified EmoBank regression baseline uses the official upstream
train/dev/test split (8062 / 1000 / 1000). On the untouched test split:

| dimension | MAE | train-mean baseline MAE | Spearman rho |
| --- | ---: | ---: | ---: |
| Valence | 0.2125 | 0.2449 | 0.5460 |
| Arousal | 0.1782 | 0.1904 | 0.3120 |
| Dominance | 0.1494 | 0.1563 | 0.2573 |

The regressor beats the train-mean baseline on MAE for all three dimensions,
but Arousal and Dominance rank correlations are still modest. The machine-readable
record is `eval/results/emobank_affect_baseline.json`.

### Dialogue-signal baseline

The learned dialogue-signal tracker was verified on the pinned XDailyDialog
English train/dev/test files (83035 / 8025 / 7716 utterances). On the untouched
test file:

| task | accuracy | macro-F1 | majority accuracy |
| --- | ---: | ---: | ---: |
| dialogue act | 0.8025 | 0.7274 | 0.4557 |
| basic emotion | 0.8249 | 0.4310 | 0.8166 |

The dialogue-act signal is substantially above the majority baseline. The
emotion signal is much weaker: macro-F1 is only 0.431 and accuracy is only
slightly above the majority baseline on test (and below it on dev). EQ-Layer
therefore treats emotion as supporting evidence rather than standalone proof of
subtext. The machine-readable result is
`eval/results/xdailydialog_signal_baseline.json`.

### Higher-level subtext audits

Direct higher-level supervision was investigated rather than assumed:

| source / target | verified result | routing decision |
| --- | --- | --- |
| DialogBank correction | 4 direct examples | **NO-GO** — insufficient supervision |
| DialogBank disagreement | 1 direct example | **NO-GO** — insufficient supervision |
| Coarse Discourse disagreement | test ROC-AUC 0.8111, AP 0.1420, F1 0.2278 | **NO-GO** — ranking signal exists, but precision/recall are insufficient for `challenge` routing |
| DBDC3 hard breakdown | test ROC-AUC 0.5057, AP 0.2555 | **NO-GO** — effectively chance-level text-only generalisation |

These results are preserved rather than optimized away. The corresponding
machine-readable records are
`eval/results/dialogbank_label_inventory.json`,
`eval/results/coarse_disagreement_no_go.json`, and
`eval/results/dbdc3_breakdown_no_go.json`.


## Evidence provenance

A subtext label is not enough by itself. `SubtextDecision.evidence_level`
records how the label was obtained:

- `verified` — caller/external annotation supplied the state.
- `direct_learned` — the upstream training target directly matches the emitted label.
- `derived` — multiple learned signals are composed into a higher-level control label.
- `structural` — explicit conversational form or marker triggered the label.
- `fallback` — learned evidence was too weak and an auditable fallback was used.

For example, XDailyDialog directly supervises `question`, but it does not
directly supervise EQ-Layer's `challenge` or `exhaustion`; those remain
`derived`. `StanceDecision` uses the same principle: factual
`user_right/user_wrong` stays `unknown` unless externally verified.

## Blind response-level evaluation

Component metrics do not establish that EQ-Layer improves the final response.
The repository therefore includes a condition-blind paired human-evaluation
harness. The complete pre-registered protocol is in `eval/AB_PROTOCOL.md`.

Generate paired responses with the **same base model** in both arms:

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

The configured command receives JSON on stdin and returns either plain response
text or JSON containing `text`/`response`. Secrets belong in environment
variables, not command arguments. `eval/cases.jsonl` is rejected by default
because it is a development set, and gold annotations are ignored unless an
explicit oracle-analysis flag is supplied.


Prepare pairs as JSONL:

```json
{"id":"case-001","context":[{"role":"user","content":"..."}],"baseline":"...","eq":"..."}
```

Blind and randomize them:

```bash
python eval/ab_prepare.py pairs.jsonl ballot.jsonl key.json --seed 42
```

Keep `key.json` away from raters. For every case, rate A/B/tie on:

- `intent_fidelity`
- `appropriateness`
- `actionability`
- `non_patronizing`
- `overall`

Then score a single rater:

```bash
python eval/ab_score.py rated_ballot.jsonl key.json
```

Or score multiple raters on the same blinded cases:

```bash
python eval/ab_score_multi.py key.json rater1.jsonl rater2.jsonl rater3.jsonl
```

The single-rater scorer reports EQ wins/losses/ties, non-tie win rate, Wilson
95% confidence intervals, and an exact two-sided sign test. The multi-rater
scorer additionally aggregates **case-level majority preference** and reports
**Fleiss' kappa** for inter-rater agreement; it does not treat every rater vote
as an independent case.

The harness does **not** manufacture a result: response generation and human
ratings must come from a real pre-registered comparison.

## Selection

Policies declare preconditions, and the most constrained applicable policy
wins. There is no first-match rule, so adding a policy cannot silently change
what a looser one does to a state you did not think about.

Two consequences worth arguing about:

- **Implicit tie-breaking is forbidden.** Selection ranks by specificity, then
  explicit `priority`. If two applicable policies still tie, the selector
  raises an ambiguity error instead of silently picking whichever was declared
  first.
- **`stance` is not guessed.** Whether the user is right is a semantic
  judgement, so the tracker returns `unknown` unless a case or caller supplies
  one. Policies that need it declare `user_is_right` or `stance_unknown`, so
  "I could not tell" stays visible in the output instead of becoming a
  confident wrong answer.

Cases can carry `annotated` fields (`stance`, `subtext`) for signals that
keywords genuinely cannot recover — exhaustion is the current example.

## Layout

```
eq_layer/
  policies.py   joint affect/intent policy taxonomy + selector
  affect.py           affect-state tracker interface + zero-dependency structural fallback
  trained_affect.py   learned VAD regressor + multi-turn affect adapter
  trained_subtext.py     learned dialogue-act/emotion signals + conservative subtext derivation
  trained_disagreement.py experimental direct-disagreement model; verified no-go for routing
  trained_breakdown.py   experimental text-only breakdown model; verified no-go for routing
  tracker.py             composes affect, subtext and stance into policy state
  intent.py           intent state + zero-dependency fallback
  trained_intent.py   learned TF-IDF + logistic-regression adapter
  data/intent_train.jsonl  bundled training corpus
  steer.py               policy + intent → decode path, and the scorer
  response_experiment.py same-model baseline-vs-EQ generation engine
eval/
  cases.jsonl          development cases, not final A/B evidence
  AB_PROTOCOL.md       pre-registered response-level evaluation protocol
  ab_generate.py       paired response generation
  ab_prepare.py        A/B blinding
  ab_score.py          single-rater scoring
  ab_score_multi.py    multi-rater scoring
  run.py
```

## Scoring

Three metrics, deliberately narrow:

- **specificity** — does the response name the concrete thing the user actually said, or a category of difficulty?
- **genericness** — count of empty validation phrases. Penalised on purpose: these are cheap under human raters, so models learn to produce them unless a cost is attached.
- **escalation latency** — how many turns pass after tension rises before the register changes.

If genericness is not penalised, a model will score well by saying nothing in particular.

## Affect and conversational signals are now partly learned

`TrainedAffect` learns continuous Valence, Arousal and Dominance from EmoBank,
then converts Valence to `[-1, 1]` and Arousal to `[0, 1]` for policy state.
Multi-turn `escalation_delta` is computed from consecutive learned arousal
predictions rather than keyword counts.

`TrainedSubtext` separately learns dialogue-act and basic-emotion signals from
XDailyDialog. Those supervised signals are then combined with V/A state to
derive higher-level control labels conservatively:

- learned `question` -> `question`
- learned `directive` + independent heat -> `demand`
- learned `question` + anger/disgust + high arousal -> `challenge`
- learned sadness + negative low-arousal V/A -> `exhaustion` / `resignation`

`correction` and disclosure boundaries remain structural because the chosen
upstream data does not directly supervise those categories.

`stance=user_right/user_wrong` is **not** inferred from generic dialogue.
Whether a user is factually or procedurally correct requires external evidence.
`AnnotationStanceResolver` therefore emits `unknown` unless a verifier or
annotation supplies a stance. This is intentional rather than a missing
confidence threshold.

The remaining limitations are:

- **`escalating_past_n` excludes the current turn.** The question is whether
  the preceding turns were rising, so one polite message after three hostile
  ones does not reset the register.
- **`question` vs `challenge`** turns on whether escalation markers are
  present, not on punctuation. `Sözleşme kaç gün geçerli?` is a question;
  `Sen de mi?!` is a challenge.
- **Exhaustion/resignation still lacks direct supervision.** The current label
  is a `derived` composition of learned sadness plus low-valence/low-arousal
  state. It must not be described as a directly learned exhaustion classifier.
- **First-word matching strips punctuation.** `Neden?` must match `neden` or
  the one-word repeated question — the exact shape `escalating` exists to
  catch — silently never fires.

The remaining hard gap is direct, transferable supervision for higher-level
states such as correction, exhaustion/resignation, and factual stance. Two
plausible direct proxies — disagreement detection and text-only breakdown
detection — were tested and retained as no-go results instead of being wired
into policy routing. The tracker exposes where labels are direct, derived,
structural, fallback, or externally verified.

## Contributing

Cases are the main entry point. Open one for a failure mode you can describe concretely — especially the ones where two policies seem equally right.

Policy changes go through an RFC first: `.github/RFC_TEMPLATE.md`.

## License

MIT. See `LICENSE`.