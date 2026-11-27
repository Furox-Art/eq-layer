# EQ-Layer

EQ-Layer is an early-stage, model-agnostic control layer that gives a language
model an explicit decision about how to respond before it responds. It tracks
affect, intent belief, repair state, and interaction quality across turns,
selects a response policy, and compiles that policy into factored controls that
are rendered as structured prompt instructions (a system instruction plus a
prompt prefix). Steering is prompt-level: the `Steer.logit_bias` field is an
unused placeholder, and no token- or decode-time intervention is implemented.
The intermediate decisions stay inspectable instead of being hidden inside an
opaque generation step.

The project does not retrain the model and does not claim that affect-aware
dialogue policy, intent routing, or repair are new. Its narrower question is
whether a transparent control layer can improve a fixed general-purpose model
while keeping each decision auditable. See
[Prior art and novelty](PRIOR_ART_AND_NOVELTY.md) for the historical map and
the claims this project deliberately avoids.

## Status

Early. The evaluation harness runs, but its numbers are development signals,
not a real-world EQ benchmark. There is no standard benchmark for EQ; `eval/`
is a first attempt at measuring one. The claim that the layer improves a fixed
model's responses is untested: the pre-registered human A/B evaluation in
`eval/AB_PROTOCOL.md` is pending, with no rated results.

## Documentation

- [Prior art and novelty](PRIOR_ART_AND_NOVELTY.md) — the research boundary and
  relevant pre-LLM work.
- [A/B latency and quality benchmark](AB_LATENCY_BENCHMARK.md) — what the layer
  costs and what it decides, measured with the layer enabled and disabled.
- [Publishing](PUBLISHING.md) — how the PyPI package and npm launcher are
  versioned and released.

## Install

```bash
pip install "eq-layer[ml]"
```

The zero-dependency CLI route needs no trained artifacts:

```bash
python -m eq_layer.cli route --intent-mode heuristic --pretty
```

The full learned pipeline requires `.joblib` artifacts produced by the training
commands in the repository README; those artifacts are not shipped.
