# Intent loss-matrix calibration protocol

Status: **preregistered scaffold; no calibrated matrix exists yet**.

The production matrix remains `hand-specified-v1` until an independent
calibration study passes this protocol. The 120-case A/B held-out set, its
generated responses, blinded preference outcomes, and any prior sensitivity
audit are **forbidden tuning data**.

## Target estimand

Estimate the relative interaction cost of each response decision given an
independently adjudicated user intent:

- decisions: `action_request`, `status_check`, `explanation`,
  `question`, `statement`, `clarify`;
- true intents: `action_request`, `status_check`, `explanation`,
  `question`, `statement`, `unknown`.

Costs are decision losses, not probabilities and not response-quality scores.

## Data separation

Before any cost ratings are collected:

1. Freeze the calibration case file and record its SHA-256.
2. Reject every exact `case_id` or upstream `source_id` that appears in the
   A/B held-out set.
3. Do not use intent-training examples as calibration cases.
4. Do not inspect blinded A/B response preferences while calibrating.
5. Keep a second, frozen validation split that is not used to estimate costs.

The calibration and A/B response-quality experiments answer different
questions and must remain separate.

## Annotation unit

Each rater sees the dialogue context plus six abstract response moves. They do
**not** see EQ-Layer's current matrix, router choice, or model-generated A/B
responses.

For every case, raters provide:

- one adjudicated `true_intent`;
- an interaction-cost rating from 0 (no mismatch cost) to 4 (severe mismatch)
  for every candidate decision.

Use at least three independent raters per retained case. Cases with unresolved
intent disagreement are retained as disagreement records but are excluded from
matrix estimation unless the preregistered adjudication rule resolves them.

## Estimation

The reference implementation in `eval/loss_matrix_calibration.py`:

- validates complete decision ratings;
- rejects duplicate rater/case pairs;
- enforces consistent adjudicated intent per retained case;
- can reject overlap with forbidden A/B case/source IDs;
- averages raw 0-4 costs within each decision × true-intent cell;
- scales by 4 to produce a transparent 0-1 candidate matrix;
- does **not** force diagonal cells to zero;
- does **not** promote the candidate to production.

Do not search cost values to maximize the blinded A/B preference rate.

## Validation gate

A candidate matrix may be considered for a new version only after it is frozen
and evaluated on the independent validation split. Report, at minimum:

- cases and raters per true-intent class;
- cell means and uncertainty intervals;
- inter-rater agreement;
- routing changes relative to `hand-specified-v1`;
- clarification rate by stratum;
- failure/repair strata separately;
- sensitivity to reasonable perturbations.

A candidate that materially reduces one error class while causing a large,
unexplained deterioration elsewhere is a mixed result, not a pass.

## Claim boundary

Until this protocol has independent human data and the validation gate passes,
the correct statement is:

> The current loss matrix is hand specified and sensitivity-audited; it is not
> empirically calibrated or known to be optimal.
