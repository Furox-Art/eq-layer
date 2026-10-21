# Security Policy

EQ-Layer decides how a model should speak: it classifies intent, affect and
repair, selects a response policy, compiles a factored action, steers the decode
path, and audits the response against inspectable contracts. None of that is
sandboxing. This page says what actually matters for security, how to report a
problem, and what is out of scope.

## What the layer touches

Understanding the attack surface means not conflating these:

| Component | Runtime dependency | Security-relevant because |
|---|---|---|
| `eq_layer` library | None (optional `scikit-learn` for trained models) | Pure computation. It reads messages and emits decisions and text; it makes no network calls and runs no user-supplied code. |
| Trained artifacts (`.joblib`) | `joblib` | `joblib` is pickle-backed. `TrainedAffect.load` / `TrainedIntent.load` / `TrainedSubtext.load` on an **untrusted** file is code execution, not data loading. |
| Steering + auditor | Host SDK | The layer injects `system_instruction` / prompt rewrites into your generation call. It cannot constrain what the host then does with the model output. |
| npm package (`eq-layer`) | Node >= 22.14 | `install` copies the Agent Skill into discovery roots (`~/.claude/skills/…`, `~/.codex/skills/…`, …) and `--force` removes an existing target directory first. `route` forwards arguments to `python -m eq_layer.cli`. |
| PyPI package (`eq-layer`) | Python >= 3.10 | Same library surface; adds the `eq-layer` CLI entry point. |

The honest summary: the library does not execute anything you hand it except
model artifacts, and those are exactly the part you must trust.

## Supported versions

| Version | Supported | Notes |
|---|---|---|
| `0.2.x` on npm | Yes | `0.2.5` is the current release. |
| `0.2.x` on PyPI | Yes | `0.2.5` is the current release, published with a PEP 740 attestation. |
| `<= 0.1.x` | No | No fixes. Reproduce on `main` and open an issue if it still reproduces. |

Only the current release is exercised by CI (`.github/workflows/eval.yml` and the
release workflows). Earlier versions are not.

## Verifying an install

Two channels, three mechanisms. Conflating them makes a supply-chain check prove
less than it looks:

| Mechanism | What it proves | Where to read it |
|---|---|---|
| `dist.integrity` / sha256 digest | The bytes you received are the bytes that were published. Integrity only. | `npm view eq-layer dist.integrity`; PyPI `digests.sha256` |
| npm `dist.signatures` | The **registry** signed the packument entry. A transport signature, not build provenance. | `npm view eq-layer dist.signatures` |
| Build attestation | A CI run built this artifact from a named repo, workflow and commit. | see below |

- **PyPI — attested.** Publishing goes through `pypa/gh-action-pypi-publish`
  with `permissions: id-token: write`, so every release artifact carries a
  [PEP 740](https://peps.python.org/pep-0740/) attestation:

  ```bash
  curl -s https://pypi.org/integrity/eq-layer/0.2.5/eq_layer-0.2.5-py3-none-any.whl/provenance
  ```

  Note the URL shape: attestations are served **per file**. A `404` on this URL
  means "no such endpoint", not "no attestation".

- **npm — published with `--provenance` on the default OIDC path.**
  `.github/workflows/publish-npm.yml` runs in GitHub Actions with
  `id-token: write`, which lets the npm CLI mint a Sigstore identity from the
  workflow rather than from the registry. Releases published through the
  explicit `use_token_fallback=true` dispatch carry no attestation.
  Verify a default release with:

  ```bash
  npm audit signatures eq-layer
  ```

  `npm audit signatures` covers registry signatures. To fetch the attestation
  bundle itself:

  ```bash
  curl -s https://registry.npmjs.org/-/npm/v1/attestations/eq-layer@0.2.5 \
    | head -c 500
  ```

  If that returns `404`, verify by digest instead
  (`npm view eq-layer@0.2.5 dist.integrity`) and treat that as integrity only.

## Reporting a vulnerability

**Use GitHub's private vulnerability reporting.** It opens a private thread
between you and the maintainer, and nothing becomes public until an advisory is
published.

<https://github.com/Furox-Art/eq-layer/security/advisories/new>

Please do not open a public issue for a suspected vulnerability — that publishes
a timeline for an unfixed problem. If private reporting is unavailable to you,
open a regular issue containing only enough detail to ask for a private
channel, and no exploit details.

Include, where you can:

- the version and registry (npm or PyPI) you installed from,
- operating system and Python/Node version,
- a minimal transcript, artifact file, or CLI invocation that reproduces it,
- what an attacker gains, and what they need in order to do it.

## What to expect

| Stage | Target |
|---|---|
| Acknowledgement | 48 hours |
| First assessment, with severity and a fix or a mitigation | 7 days |
| Fix released | Case by case; security fixes are prioritised over feature work |

These are targets, not a contract. Out-of-scope reports get that answer at the
first assessment. There is no PGP key for this project, so reports are not
encrypted beyond HTTPS to GitHub; say so if that matters and an encrypted
channel will be arranged.

## In scope

- **Untrusted model artifacts.** `joblib.load` reaching arbitrary code
  execution from a crafted `.joblib` — either a fix that hardens the load path
  or documentation that makes the trust boundary explicit, which is what ships
  today.
- **Structural auditor false negatives.** A response that passes `audit()`
  while violating a hard contract: question budget overruns, required
  clarification never issued, unsupported privacy promises ("I'll keep this
  just between us"), unknown-stance admissions presented as fact, repair
  repetition, low-verbosity overruns.
- **Steering or policy selection that drops a boundary.** The `boundary`
  policy, or an explicit user constraint, being overridden by affect-driven
  adaptation when the factually requested task is unchanged.
- **Installer path handling.** `npm` `install` (or `tools/install_agent_skills.py`)
  writing, deleting or following paths outside the intended target root — for
  example a target name or `--home` value that escapes the discovery directory,
  or `--force` removing something other than the skill directory.
- **CLI argument forwarding.** `eq-layer route` arguments reaching the spawned
  `python -m eq_layer.cli` in a form that is not what the user typed.
- **Supply chain.** A published package version shipping skill content that
  does not match this repository, since installed skills are instructions that
  agents later follow.

## Out of scope

- **Not a sandbox.** The layer and the host agent run as the same user. A
  hostile process running as that user can do anything the layer can.
- **Not a truth checker.** The auditor checks structural, inspectable
  properties of a response. It does not verify facts, and a factually wrong
  response that satisfies every contract is by design not a security finding.
- **Host model behaviour.** What the wrapped model does with the steering
  text, including prompt-injection resistance of the model itself, belongs to
  the host/s SDK.
- **Dependency CVEs in unrelated tooling.** The library has no runtime
  dependencies; `scikit-learn` is optional and only used by trained adapters.
- **Dataset licensing.** Covered in `THIRD_PARTY_DATA.md`.

## Loading artifacts safely

`TrainedAffect.load`, `TrainedIntent.load` and `TrainedSubtext.load` deserialize
whatever path you give them. There is no allowlist of sources and no sandbox.
Only load `.joblib` artifacts you built yourself with `eval/` or
`tools/train_*.py`, or that you can trace to a published release with the
digest checks above. Treat a model artifact you did not build exactly like a
script you did not read.
