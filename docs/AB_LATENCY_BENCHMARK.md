# A/B latency and quality benchmark

Component metrics do not say what the layer costs. This page reports a paired
A/B run of the **same fixed set of user requests** with the EQ layer
**enabled** and **disabled**, measured on one machine, so the added latency
and the structural quality of the decision can be compared directly.

Machine-readable record: `eval/results/ab-latency.json`.

## What is and is not measured

There is no LLM in the measured path. The layer emits a control decision
(`task_move` / `social_move` / `repair_move` / `realization`) that is injected
into the decode path. Producing the tokens is the model's job, and it is not
part of what this layer pays for. What is measured is:

- the layer's own deterministic compute (classification + routing);
- the structural quality of the decision it produces.

Consequently these numbers answer **"what does the layer cost and what does it
decide"**, not **"does the response read better"**. End-to-end response
quality needs a paired human or model-level evaluation; that protocol is in
`eval/AB_PROTOCOL.md` and its primary endpoint is the routing decision, not
response preference.

## Methodology

| property | value |
| --- | --- |
| requests | the 15 seeded cases in `eval/cases.jsonl`, unchanged in both arms |
| holdout requests | the 36 single-turn cases in `eval/intent_holdout.jsonl` |
| runs per arm | 100 passes over the full case set |
| measurements per arm | 1,500 (15 cases x 100 runs) |
| ordering | layer-on and layer-off measured back-to-back per run |
| warmup | 2 unrecorded passes per arm |
| seeds | none needed; the layer is deterministic, no sampling |
| classification vs routing | timed as two independent spans, never lumped |
| classification split | intent inference vs affect/repair/interaction/reference |
| mode | offline/deterministic, no LLM API key required |
| environment | Python 3.14.7, Windows 11 (AMD64), single run, no CI |

Reproduce:

```bash
python eval/ab_latency_eval.py
# or with an explicit run count and output path
python eval/ab_latency_eval.py --runs 100 --output eval/results/ab-latency.json
```

The learned adapter (scikit-learn) is used when installed; the
zero-dependency heuristic path is measured either way, so the benchmark runs
with or without the `ml` extra. Both adapters are reported below.

## Added latency

The layer-off arm only performs the transcript handoff a caller does in both
arms, so the delta isolates the layer's own work. Layer-off is ~0.001 ms and
is reported for completeness, not as a meaningful baseline.

### Learned intent adapter (`trained-tfidf-logreg`, bundled corpus)

| quantity | mean (ms) | median (ms) | p50 (ms) | p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| layer on (total) | 1.1992 | 1.1628 | 1.1627 | 1.4642 |
| layer off (total) | 0.0011 | 0.0010 | 0.0010 | 0.0022 |
| **added** | **1.1981** | **1.1613** | **1.1612** | **1.4636** |
| classification | 1.1733 | 1.1370 | 1.1370 | 1.4305 |
| &nbsp;&nbsp;- intent inference | 1.0751 | 1.0386 | 1.0385 | 1.3386 |
| &nbsp;&nbsp;- affect / repair / interaction / reference | 0.0982 | 0.0714 | 0.0713 | 0.1953 |
| routing (selector + factored action) | 0.0255 | 0.0245 | 0.0245 | 0.0327 |

### Zero-dependency intent adapter (`heuristic`)

| quantity | mean (ms) | median (ms) | p50 (ms) | p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| layer on (total) | 0.1266 | 0.1111 | 0.1110 | 0.2274 |
| layer off (total) | 0.0006 | 0.0006 | 0.0006 | 0.0011 |
| **added** | **0.1259** | **0.1105** | **0.1104** | **0.2265** |
| classification | 0.1090 | 0.0902 | 0.0902 | 0.2087 |
| &nbsp;&nbsp;- intent inference | 0.0271 | 0.0287 | 0.0287 | 0.0458 |
| &nbsp;&nbsp;- affect / repair / interaction / reference | 0.0819 | 0.0555 | 0.0554 | 0.1771 |
| routing (selector + factored action) | 0.0171 | 0.0161 | 0.0161 | 0.0220 |

Reading the split:

- with the learned adapter, **intent inference is ~92% of classification and
  ~90% of the whole added cost**; routing is ~2%.
- with the heuristic adapter the balance moves: **affect/repair state becomes
  the larger half (~75% of classification)**, because the learned TF-IDF
  classifier is what the heuristic adapter replaces.
- routing is the cheapest part of the layer in both modes, at ~0.02-0.03 ms.

## Quality

Quality is script-computed on the **control decision**, not on generated
prose, because the harness has no model to generate prose. Both arms are
scored by the same script; a layer-off request has no decision, so its
baseline is the empty default and the deltas are real measured differences.

Metrics, fixed before the run:

- **intent-label accuracy** — predicted `intent_kind` against the labelled
  `expected_intent`.
- **appropriateness rubric** (8 checks) — task/social/repair move present,
  realization present, question budget within limit, `no_guess` when
  clarifying, boundary keeps warmth low, register known.
- **communication quality** (6 checks) — verbosity/directness/warmth in
  range, question budget consistent, `scope_limited`/`no_guess` boolean.

| metric | learned on | learned off | learned delta | heuristic on | heuristic off | heuristic delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| intent accuracy (4 labelled cases) | 1.000 | 0.000 | +1.000 | 1.000 | 0.000 | +1.000 |
| intent holdout accuracy (36 cases) | 1.000 | - | - | 0.278 | - | - |
| appropriateness rubric (15 cases) | 0.992 | 0.125 | +0.867 | 1.000 | 0.125 | +0.875 |
| communication quality (15 cases) | 1.000 | 0.167 | +0.833 | 1.000 | 0.167 | +0.833 |

Two things the numbers show that a single headline would hide:

- **The learned adapter and the heuristic adapter are not interchangeable.**
  Both reach 1.000 intent accuracy on the 4 labelled seed cases, but on the
  36-case holdout the learned adapter scores **1.000** and the heuristic
  adapter **0.278**. The learned path is where intent understanding lives;
  the heuristic path is a fallback, not a peer.
- **The layer-on appropriateness score is 0.992, not 1.000, with the learned
  adapter.** One case fails a rubric check. The harness reports this rather
  than rounding it away, and the invariant check only requires the layer-on
  arm to be a real decision (> 0.5), not a perfect one.

## Per-request added latency (learned adapter, mean ms)

| case | added (ms) | | case | added (ms) |
| --- | ---: | --- | --- | ---: |
| case-001 | 1.2074 | | case-009 | 1.2991 |
| case-002 | 1.2705 | | case-010 | 1.1415 |
| case-003 | 1.1914 | | case-011 | 1.3188 |
| case-004 | 1.2958 | | case-012 | 1.1344 |
| case-005 | 1.2053 | | case-013 | 1.1346 |
| case-006 | 1.2021 | | case-014 | 1.1825 |
| case-007 | 1.1073 | | case-015 | 1.0992 |
| case-008 | 1.1986 | | | |

No case is an outlier: the spread is 1.10-1.32 ms, roughly +/-9% around the
mean, so the cost is uniform across one-turn and multi-turn transcripts
rather than driven by a few long conversations.

## Run-to-run variation

Latency numbers are machine- and load-dependent. Three consecutive runs of
this benchmark on this machine gave:

| run | learned added p50 (ms) | learned added p95 (ms) | heuristic added p50 (ms) |
| --- | ---: | ---: | ---: |
| run A | 1.2632 | 1.6338 | 0.0886 |
| run B | 1.1479 | 1.4864 | 0.0895 |
| run C (committed) | 1.1612 | 1.4636 | 0.1104 |

The heuristic path is stable to within a few percent, and the learned path
moved ~9% across runs minutes apart on the same machine. The learned
latencies above should therefore be read as **approximately 1.2 ms median,
~1.5-2.0 ms p95** rather than as exact constants. The quality numbers are
identical across every run because the layer is deterministic.

## Claim boundary

- This is a **development benchmark on a 15-case seeded set**, not an
  external or human-labelled standard.
- Latency measures the **deterministic layer only**. It excludes token
  generation, network, and any model call, in both arms.
- Quality measures the **control decision**, not response quality. A perfect
  decision can still be realised poorly by a model, and these numbers cannot
  see that.
- The quality deltas are large because layer-off has no decision to score.
  They demonstrate the layer produces a decision and that the decision
  carries the required structural properties; they do **not** demonstrate the
  layer improves user-perceived response quality.
- The 4 labelled seed cases are a small denominator; the 36-case holdout is
  the more meaningful intent signal, and only the learned adapter scores well
  on it.
