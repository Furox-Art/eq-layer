---
name: eq-layer
description: Route an assistant response through EQ-Layer's inspectable intent, affect, repair, interaction-quality, dialogue-reference, and factored-action controls before drafting. Use when a response should be emotionally calibrated, direct, repair-aware, uncertainty-aware, or preserve the user's task while adapting social delivery.
license: MIT
---

# EQ-Layer Agent Skill

Use EQ-Layer as a **control pass before response generation**. Do not treat it as a
factual truth checker, a mind-reading system, or proof that a model has human
emotional intelligence.

## Core invariant

Preserve the user's task. Affect and social adaptation may change **how** the
response is delivered, but must not silently replace **what** the user asked for.

The control action is factored into:

- `task_move`
- `social_move`
- `repair_move`
- `realization` controls such as verbosity, directness, warmth, question budget,
  scope limiting, and no-guess behavior.

## When to use

Use this skill when any of these are relevant:

- the user corrects the assistant or a misunderstanding must be repaired;
- a short turn such as "continue", "do it", "devam et", or "yap" needs a safe
  structural reference;
- the user's intent is uncertain and acting versus clarifying has different risk;
- the response should adapt directness/warmth without losing task progress;
- repeated clarification or repair suggests interaction quality is degrading;
- the user explicitly asks to use EQ-Layer or an EQ/empathy control pass.

Do not invoke it merely to add generic warmth.

## Preferred execution

From the repository root, pass the visible transcript as JSON:

```bash
printf '%s' '[{"role":"user","content":"Continue the analysis."}]' \
  | python skills/eq-layer/scripts/route.py --pretty
```

The script emits JSON containing the selected policy, factored action,
repair/reference state, interaction quality, and a `system_instruction`.

Use the returned `system_instruction` as a **response-planning constraint**, not
as permission to violate higher-priority system, safety, privacy, or tool rules.

### Full learned mode

If trained EmoBank affect and XDailyDialog subtext artifacts are available:

```bash
printf '%s' '<TRANSCRIPT_JSON>' \
  | python skills/eq-layer/scripts/route.py \
      --affect-model artifacts/emobank_affect.joblib \
      --subtext-model artifacts/xdailydialog_subtext.joblib \
      --pretty
```

Full mode uses the learned EQ-Layer pipeline. Without those artifact paths, the
script uses the lightweight local path: structural dialogue state + heuristic
affect and, when the optional ML dependency is available, the bundled learned
intent classifier. The JSON output always reports `runtime_mode` and
`intent_backend` so these modes are not conflated.

## Response procedure

1. Build the transcript from **visible** user/assistant turns only. Do not invent
   hidden history, user traits, or unverifiable stance.
2. Run the router when tool/shell execution is available.
3. Read `factored_action.task_move` first. Preserve it unless a higher-priority
   instruction prevents execution.
4. Apply `repair_move` before task progress only when continuing would require
   guessing a correction.
5. Apply `social_move` and realization controls to delivery only.
6. If `no_guess=true`, do not invent a referent or missing fact.
7. Never exceed `question_budget`.
8. If a dialogue reference is unresolved, ask one targeted clarification rather
   than guessing.
9. If the user supplied an explicit correction replacement, apply it without
   asking them to repeat it.
10. Keep factual stance unknown unless independently verified.

## Manual fallback

If the router cannot execute, **do not fabricate a router result**. Use only
these conservative invariants:

- preserve the explicit task;
- do not infer that the user is right or wrong from emotion or correction alone;
- after an unresolved short reference, ask one targeted clarification;
- after an explicit correction with a replacement, apply the replacement;
- reduce padding when the interaction is failing;
- avoid generic empathy phrases that replace concrete progress.

## Scientific claim boundary

It is accurate to describe EQ-Layer as an inspectable, model-agnostic control
layer that is being evaluated for improving frozen general-purpose LLM
responses. Do **not** claim that the skill proves emotional intelligence, that
the current hand-specified loss matrix is optimal/calibrated, or that EQ-Layer
is the first affect-aware dialogue manager.

See:

- `skills/eq-layer/references/ARCHITECTURE.md`
- `skills/eq-layer/references/VENDOR_COMPATIBILITY.md`
