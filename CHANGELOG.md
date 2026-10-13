# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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

[0.2.1]: https://github.com/Furox-Art/eq-layer/releases/tag/v0.2.1
[0.2.0]: https://github.com/Furox-Art/eq-layer/releases/tag/v0.2.0
