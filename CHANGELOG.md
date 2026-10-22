# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- The response auditor folds punctuation, fullwidth and zero-width characters,
  combining marks and Turkish dotted/dotless case before matching privacy
  promises and stance admissions, and it now requires whole-word matches.
  Evasions such as `KİMSEYE-SÖYLEMEM`, `kimseye<zero-width>söylemem` or
  `You’re right` no longer pass the audit, while `söylememek` no longer matches
  `söylemem`.
- `SECURITY.md` no longer claims the package makes no network calls. It
  documents the three dataset download helpers, which hosts they contact and
  which of them verify an archive digest.

### Packaging

- The `[ml]` and `[dev]` extras declare `joblib` explicitly instead of relying
  on it arriving through `scikit-learn`.
- The sdist now ships `tests/`, `eval/`, `tools/`, `docs/`, the security policy,
  changelog and citation metadata, so the packaged source matches the claims
  the project makes about its own evaluation.

### CI

- Weekly Dependabot updates for GitHub Actions and pip, OpenSSF Scorecard
  publishing, and dependency review that fails pull requests introducing a
  high-severity advisory.

## [0.2.5] - 2026-10-21

### Security

- CLI transcripts and case files share one strict JSON validator; ambiguous or
  non-string transcripts are rejected instead of being coerced (`a59f5ed`,
  `5e9ad6d`).
- Human evaluation ballots must be unique and unambiguous, and fabricated
  blinded responses or empty rater cohorts are refused (`6a198c2`, `8e66cbe`).
- Open CodeQL code-scanning alerts are resolved without behaviour changes
  (`d039add`).

### Fixed

- `tools/crosscheck.js` runs under ESM, and the SECURITY/CITATION metadata is
  repaired (`ebd3ca4`).

### CI

- Python branch coverage is enforced at 100% with zero skipped tests
  (`4ec663a`); pip dependencies are cached in the EQ validation and coverage
  jobs (`2517ece`).
- Actions are pinned to commit SHAs and CodeQL analysis runs in CI
  (`33f9a06`, `d039add`).

### Docs

- npm releases default to trusted publishing over OIDC and request provenance.
  Token mode runs only when a maintainer dispatches `publish-npm.yml` with
  `use_token_fallback=true`; an `NPM_TOKEN` secret does not select it
  (`5315bb3`).
- `create-release.yml` explicitly dispatches both publishers; a release tag
  alone does not publish npm.
- README documents `npm install -g eq-layer` / `npx eq-layer --help`, states
  that `artifacts/*.joblib` are not shipped and must be trained locally (or the
  zero-dependency route used), and publishing docs are corrected.
- Version references are updated to 0.2.5 in README, SECURITY, CITATION,
  SKILL.md and the plugin manifests.

## [0.2.2] - 2026-10-12
### Housekeeping — audit hardening

- Closes the findings of the full-repository audit: patched the concrete
  defects (correctness, fail-closed behaviour, platform independence),
  wired every standalone guard into CI, removed dead or duplicated code
  paths, and pinned every doc claim to a test so it cannot silently
  drift again. Regression tests fail without the corresponding fix.

## [0.2.3] - 2026-10-13

Second-round hardening: optional dependencies (sklearn/joblib) raise a typed
`OptionalDependencyError` with the install hint instead of a raw traceback, the
CLI maps it to exit 2 and `doctor` names the extra; `from_bundled()` is
process-cached (repeated routing no longer re-fits); trained-model state loaders
validate their schema and report the offending field; version manifests are
pinned by a parity test; PUBLISHING.md documents both npm credential modes.

## [0.2.4] - 2026-10-15

Deterministic-core hardening:

- Golden-decision contract: 25-row fixture covering every policy family,
  register, intent mode and intent kind, asserting the FULL decision object;
  deterministic across announcement order, locale, PYTHONHASHSEED and encoding
  (a fresh interpreter re-runs the golden table).
- Classifier determinism: repeated invocation is byte-identical (stubbed
  estimators when sklearn/joblib are absent).
- Fallback contract: 14 pinned fallback decision objects plus a differential
  test proving fallback == routed decision when the model agrees.
- `response_experiment` is reproducible: the only randomness (control-arm
  shuffle) and the timestamp are injectable, so the same experiment + fixture
  yields an identical verdict.
- Five real defects fixed by those tests: arousal was allowed below zero
  (outside the [0,1] scale), blank user turns counted as a three-turn
  interrogation run, README's policy-family list contradicted the register
  (this was failing on origin/main), CITATION.cff version drift.
- Docs honesty: a guard test pins README's claims to the real code.

Tests: 272 passed (106 baseline + 166 new).

## [Unreleased]

### Changed

- **Routing decisions are now pinned by a golden table.** `tests/fixtures/route_golden.jsonl`
  stores 25 fixed (messages, intent mode, full decision object) cases covering every
  policy family, register and intent kind the route paths can emit, and
  `tests/test_route_golden.py` asserts each one against `route`,
  `_lightweight_route` and the decision core. The table also pins that selection is
  independent of the order the policies are declared in, that repeated routing is
  byte-identical, and that a fresh interpreter under a different `PYTHONHASHSEED`,
  locale and encoding still produces the identical decision.

- **Fallback contract pinned for 14 message sets.** The lightweight route and the full
  learned route are now differentially tested: when a learned component's decision
  agrees with the structural one, the two paths must emit the identical factored action,
  steering instruction, repair/reference/interaction-quality state, stance and subtext.
  A deliberately disagreeing stub is asserted to override that agreement, so the equality
  cannot pass vacuously.

- **Classifier determinism.** Repeat invocation and independently re-loaded stub
  artifacts now have to produce byte-identical output for the intent posterior, the
  learned affect state and the full learned route. The stubs are fitted-estimator
  stand-ins loaded through the real artifact path, so the learned branches are
  exercised without scikit-learn.

- **Reproducibility of the recycling harness.** `generate_pairs` now quarantines its
  randomness behind an injectable `rng` (still seeded from `seed`) and its timestamps
  behind an injectable `clock` whose default is a fixed epoch, so the same experiment
  plus the same fixture serialises to identical bytes. Each record carries
  `generation.generated_at_epoch` from that clock.

- **The README's response-policy families are listed exhaustively.** The list stopped at
  `boundary` with an ellipsis; it now names all eight registers and a doc-guard test
  pins the list to `Register`, along with the intent labels, the bundled-corpus labels,
  the constraint names, the evidence levels, the layout paths and the skill's factored
  action and runtime-mode claims.

### Fixed

- **A whitespace-only user turn read as a three-turn interrogation run.**
  `HeuristicAffect._escalating` counted blank turns towards the three-consecutive
  wh-word shape while checking only the non-blank ones, so two wh-word turns plus a
  blank one were classified as escalating. It now filters blank turns the same way
  `_demand_pattern` does; a genuine three-turn run still fires.

- **A softening-only turn produced a negative arousal.** `_arousal` floored only the
  upper end, so a turn made of softeners (galiba/aslında/belki) reported an arousal
  below zero on a `[0, 1]` scale and handed the policy state an impossible value.

- **`CITATION.cff` stopped declaring its version.** The 0.2.3 release commit dropped
  the quotes around the version field, which the version-parity test reads, so the
  suite failed on a released tag.

### Housekeeping — second-round audit detail

- **Audit M-2/M-7 — a missing or unusable `ml` extra is now a typed error.**
  Every optional import in `eq_layer/` (scikit-learn in the five trained-model
  builders, joblib in every artifact save/load) raised a bare `ImportError` whose
  install command was only free text, so a missing extra surfaced as an
  unhandled traceback at request time. All of them now raise the exported
  `OptionalDependencyError`, which still subclasses `ImportError` (so the CLI's
  heuristic fallback keeps working) but names the package and the install
  command.   `eq-layer route --intent-mode learned` exits with code 2 and that
  message on stderr instead of a traceback, and `eq-layer doctor` prints the
  install hint. The same mapping now covers the `ImportError` joblib raises
  while rebuilding an artifact fitted on a machine that had the extra.

- **Audit M-5 — the bundled intent adapter is no longer re-fitted per call.**
  `TrainedIntent.from_bundled()` rebuilt and refit the classifier on every call
  (~2.3 s each), so any request path that loads the bundled adapter paid the
  full fit per request. The fit is now memoised per process on (artifact path,
  artifact mtime, both thresholds), making repeated calls O(1) and refilling the
  memo only when the corpus on disk changes.

- **Audit H-2 — malformed trained-model state is rejected at load time.** The
  artifact loaders indexed the persisted payload directly, so a missing key
  escaped as a raw `KeyError`, a non-mapping payload as a `TypeError` from the
  dataclass constructor, and a wrongly typed threshold was silently coerced by
  `float()`. The schema is validated first and a typed `ModelStateError` names
  the model, the offending field and the artifact path.

- **Audit D-1 — one source of truth for the version, now guarded by a test.**
  `pyproject.toml`, `package.json`, `plugin.json`, `.codex-plugin/plugin.json`,
  `CITATION.cff` and `eq_layer/_version.py` all report `0.2.2` (this repository
  has no `package-lock.json`). A new test pins them together so a future bump
  cannot drift across the shipped manifests.

- **Audit docs — `docs/PUBLISHING.md` described only the first-publication npm
  token flow** and still named `0.2.1` as the current release. Both credential
  modes of `publish-npm.yml` are documented now (token mode with
  `--no-provenance`, OIDC trusted publishing with `--provenance`), with the
  `NPM_TOKEN` secret as the switch, and `README.md` carries the same summary.

- **Audit M4 — one source of truth for the version string.** The Python side
  no longer carries hardcoded copies: `eq_layer/_version.py` resolves the
  version through `importlib.metadata` when installed and reads the same
  `[project] version` field from the nearest `pyproject.toml` from a source
  checkout, and `eq_layer.__version__`, `eq-layer doctor`, and the package
  export all use it. The npm launcher reads the canonical `version` field of
  `package.json` at runtime through the new `npm/version.mjs` instead of
  hardcoding it in three places. `docs/PUBLISHING.md` now documents both
  canonical locations, and the provably stale `0.2.0` claims in
  `SECURITY.md` and `docs/PUBLISHING.md` (current release, integrity and
  attestation commands) now point at `0.2.1`.

- **Audit H5 — one shared text-normalisation implementation.** `_normalise`
  was copy-pasted across five modules and had drifted into two behaviours
  (lowercase + whitespace collapse, and punctuation-to-space) with the
  dialogue-reference matcher additionally dropping the Turkish combining dot
  above. Both variants now live in the new `eq_layer.textnorm`
  (`normalise_light`, `normalise_plain`), and `auditor`, `preferences`,
  `interaction`, `intent`, and `dialogue_reference` import them instead of
  maintaining private copies. Behaviour is unchanged on canonical cases,
  including dotted/dotless-I matching, and is locked in by parity tests.

### Fixed

- **Audit C1 — persistent model commands can no longer hang an experiment.**
  The persistent model-command path blocked on `readline()` with no deadline,
  so a wedged child froze the whole experiment synchronously. The configured
  `--timeout-seconds` (default 120s) is now enforced on the persistent path
  as well: on expiry the child is force-killed and a typed
  `ModelCommandTimeout` is raised. Both command modes and the package export
  `eq_layer.ModelCommandTimeout` raise that one typed error, and
  `eval/ab_generate.py` surfaces it as a single-line error with exit code 3
  instead of an unhandled traceback.

## [0.2.1] - 2026-10-12

### Added

- `SECURITY.md` policy and `CITATION.cff` citation metadata; package metadata
  hardened across `pyproject.toml`, `package.json`, `plugin.json`,
  `.codex-plugin/plugin.json` and the npm launcher (`npm/cli.mjs`), which no
  longer carries a separately hardcoded version string in three places.

## [0.2.0] - 2026-10-08

### Added

- A/B latency and quality benchmark harness (`eval/ab_latency_eval.py`) that runs
  the same 15 seeded requests (`eval/cases.jsonl`) twice — EQ layer enabled vs
  disabled — for 100 passes per arm (1,500 measurements per arm), with two
  unrecorded warmup passes per arm.
- Classification and routing are now timed as independent spans rather than one
  lumped number, and classification is split further into intent inference vs
  affect/repair/interaction/reference state, so the expensive half is visible
  without a second run.
- Script-computed quality metrics scored through the same script in both arms:
  intent-label accuracy, an 8-check appropriateness rubric, and a 6-check
  communication-quality rubric.
- Intent holdout scoring over the 36 single-turn cases in
  `eval/intent_holdout.jsonl`.
- Committed machine-readable results at `eval/results/ab-latency.json` and a
  written report at `docs/AB_LATENCY_BENCHMARK.md`.

### Measured

Numbers from the committed run on one machine (Python 3.14.7, Windows 11, AMD64,
offline/deterministic mode, single run, no CI):

- Added latency per request, learned intent adapter
  (`trained-tfidf-logreg`, bundled corpus): **1.16 ms median, 1.46 ms p95**
  (mean 1.20 ms). Intent inference is 1.08 ms median and routing is 0.03 ms
  median, measured independently; intent inference is ~90% of the whole added
  cost, routing ~2%.
- Added latency per request, zero-dependency heuristic intent adapter:
  **0.11 ms median**, 0.23 ms p95 (mean 0.13 ms).
- Intent accuracy on the 36-case holdout: **100% (learned) vs 27.8%
  (heuristic)** — the two adapters are not interchangeable.
- Intent-label accuracy on the 4 labelled seed cases: 100% layer-on vs 0%
  layer-off.
- Appropriateness rubric: 0.992 layer-on vs 0.125 layer-off (learned adapter);
  communication quality: 1.000 vs 0.167. One case fails a rubric check with the
  learned adapter; the harness reports this rather than rounding it away.
- Run-to-run variation: the learned path moved ~9% across three consecutive runs
  minutes apart on the same machine, so learned latencies should be read as
  approximately 1.2 ms median / ~1.5-2.0 ms p95 rather than exact constants.
  Quality numbers are identical across runs because the layer is deterministic.

### Claim boundary

Documented honestly alongside the numbers:

- There is no LLM in the measured path (offline/deterministic mode). The numbers
  measure the layer's deterministic compute cost and the structural quality of
  the control decision it produces — **not** end-to-end response quality, which
  requires a paired human or model-level evaluation (`eval/AB_PROTOCOL.md`).
- Latency excludes token generation, network, and any model call, in both arms.
- This is a development benchmark on a 15-case seeded set, not an external or
  human-labelled standard.
- The layer-off arm has no decision to score, so its quality baseline is the
  empty default; the deltas demonstrate that the layer produces a decision
  carrying the required structural properties, not that it improves
  user-perceived response quality.
- Latency is load-sensitive (~9% run-to-run variation).

## [0.1.0] - 2026-10-06

First public release: a control layer that gives an LLM a *decision* about how
to speak before it speaks, instead of asking it to be more empathetic in a
prompt.

### Added

- Policy taxonomy with registers and a selector over affect, intent, task,
  repair and interaction-quality state, plus a deterministic steering compiler
  that turns the selected decision into a system instruction.
- Heuristic intent adapter and explicit intent state; intent-aware policy
  selection.
- Optional `ml` extra (`scikit-learn`): trained TF-IDF + logistic-regression
  intent adapter with held-out evaluation and cross-validation reporting.
- EmoBank-trained dimensional affect (valence/arousal) regression adapter
  with a reproducible training command and benchmark.
- XDailyDialog-trained subtext signals and a conversation tracker with
  explicit, evidence-tiered stance tracking.
- Higher-level subtext models for direct disagreement (Coarse Discourse) and
  dialogue breakdown (DBDC3), each with corpus loaders, lightweight CI
  benchmarks, and recorded no-go results where the evidence did not support
  a claim.
- Structural response audit of generated text against inspectable control
  contracts, with opt-in one-shot regeneration guidance.
- Explicit session preference state, isolated per session.
- Conservative dialogue-reference tracking, including adversarial benchmark
  cases.
- Intent belief state with a Bayes-risk router, plus grouped and cellwise
  intent risk-sensitivity audits.
- Same-model paired response experiment engine and CLI
  (`eval/ab_generate.py`): provider-neutral command adapter, persistent
  model-command mode, arm-blind backends, stratified pilot selection, and a
  held-out case preflight/validator that rejects development sets on final
  runs.
- Blind pairwise response evaluation core: ballot preparation, single- and
  multi-rater scoring with Fleiss' kappa, and a preregistered protocol
  document.
- Agent-skill packaging for Codex, Claude, Copilot, Cursor, Grok and
  Antigravity, plus PyPI and npm distributions published from GitHub Releases
  through trusted publishing.

[0.2.1]: https://github.com/Furox-Art/eq-layer/releases/tag/v0.2.1
[0.2.0]: https://github.com/Furox-Art/eq-layer/releases/tag/v0.2.0
[0.1.0]: https://github.com/Furox-Art/eq-layer/releases/tag/v0.1.0
