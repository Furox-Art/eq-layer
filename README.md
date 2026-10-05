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

Early. The harness runs; the numbers are not yet trustworthy.

There is no standard benchmark for EQ. `eval/` is a first attempt at one, not a finished measurement. If you improve the scorer, that contribution matters as much as adding a policy.

## Quick start

```python
from eq_layer import HeuristicAffect, HeuristicIntent, Selector, Steer

messages = [
    {"role": "user", "content": "Randevumu üç kez değiştirdiler"},
]
state = HeuristicAffect().infer(messages)
intent = HeuristicIntent().infer(messages)
selection = Selector().select(state, intent)
print(selection.policy.name, intent.canonical_request)

prompt = Steer.build(selection.policy, intent).apply_to_prompt(messages[-1]["content"])
```

Run the seeded cases. Exits non-zero on any mismatch or any policy that no
case can reach:

```bash
python eval/run.py
```

## Intent is a control signal, not mind-reading

The intent layer keeps a separate representation of what the user appears to
want. It records the original request inside a canonical instruction, the
response mode, confidence, and explicit constraints such as `brief`,
`scope_limited`, and `no_guess`.

The rule is conservative: if the referent is unresolved, `needs_clarification`
stays true and the selector chooses one targeted question. The layer must not
turn "do that" into a guessed task just because a likely task exists.

Affect can still override intent during sustained escalation. A clear action
request should drive the response in a calm turn; it should not erase a
multi-turn escalation signal.

## Selection

Policies declare preconditions, and the most constrained applicable policy
wins. There is no first-match rule, so adding a policy cannot silently change
what a looser one does to a state you did not think about.

Two consequences worth arguing about:

- **Ties are broken by declaration order**, which is arbitrary and a real
  weakness. If two policies of equal specificity both apply, that is a
  taxonomy problem, not a tie to be won.
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
  affect.py     affect-state tracker interface + adapters
  intent.py     intent canonicalisation + response contract
  steer.py      policy + intent → decode path, and the scorer
eval/
  cases.jsonl   seeded cases, including the hard distinctions
  run.py
```

## Scoring

Three metrics, deliberately narrow:

- **specificity** — does the response name the concrete thing the user actually said, or a category of difficulty?
- **genericness** — count of empty validation phrases. Penalised on purpose: these are cheap under human raters, so models learn to produce them unless a cost is attached.
- **escalation latency** — how many turns pass after tension rises before the register changes.

If genericness is not penalised, a model will score well by saying nothing in particular.

## Detection is the weak part

The tracker is keywords, not a classifier. It is readable on purpose — when a
signal fires you can see why without a gradient — but it is wrong in ways that
matter, and the failures are instructive:

- **`escalating_past_n` excludes the current turn.** The question is whether
  the preceding turns were rising, so one polite message after three hostile
  ones does not reset the register.
- **`question` vs `challenge`** turns on whether escalation markers are
  present, not on punctuation. `Sözleşme kaç gün geçerli?` is a question;
  `Sen de mi?!` is a challenge.
- **Exhaustion is undetectable from words.** A resigned transcript contains no
  crisis vocabulary at all. It requires annotation, and pretending otherwise
  is how this layer starts confidently wrong.
- **First-word matching strips punctuation.** `Neden?` must match `neden` or
  the one-word repeated question — the exact shape `escalating` exists to
  catch — silently never fires.

Replacing this with a trained tracker is the obvious next step, and
`AffectAdapter` is the seam for it.

## Contributing

Cases are the main entry point. Open one for a failure mode you can describe concretely — especially the ones where two policies seem equally right.

Policy changes go through an RFC first: `.github/RFC_TEMPLATE.md`.

## License

MIT. See `LICENSE`.