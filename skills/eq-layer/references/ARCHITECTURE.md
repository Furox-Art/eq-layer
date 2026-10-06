# EQ-Layer skill architecture

## Control path

```text
visible transcript
  -> affect / intent / repair / interaction-quality / reference state
  -> discrete policy
  -> factored action
  -> steering instruction
  -> unchanged base model response
```

The skill does not retrain the host model.

## Runtime modes

### lightweight-local

Designed to run without model artifacts.

- affect: `HeuristicAffect`;
- intent: bundled `TrainedIntent` when optional scikit-learn is available,
  otherwise `HeuristicIntent`;
- repair: structural;
- interaction quality: structural;
- short-reference/QUD-inspired state: structural;
- policy/action/steering: deterministic EQ-Layer code.

This mode is useful for broad portability. It is not equivalent to the full
learned research pipeline.

### full-learned

Enabled only when both trained artifact paths are supplied.

- affect: trained EmoBank adapter;
- subtext/dialogue signals: trained XDailyDialog adapter;
- intent: bundled learned intent classifier;
- repair/reference/interaction quality: structural;
- factored action and steering: EQ-Layer production path.

## Evidence discipline

Keep direct, structural, learned/derived, and externally verified evidence
separate. Repair does not establish factual correctness. Affect is not a
persistent user trait. Interaction quality is not emotion.

## Current research boundary

The 120-case same-model baseline-vs-EQ generation pipeline is reproducible and
blinded-ballot ready. Human preference adjudication remains a separate empirical
step. The production loss matrix is still `hand-specified-v1`; the repository
contains a separate independent calibration protocol and prohibits tuning it on
the same blinded A/B preference outcomes.
