# Publishing EQ-Layer

EQ-Layer publishes two artifacts with the same semantic version.

- **PyPI `eq-layer`**: the Python runtime/control layer.
- **npm `eq-layer`**: the portable Agent Skill installer and CLI bridge.

Current release: **0.2.5**.

## Version source of truth

Both packages ship the same semantic version, and each side has exactly one
canonical location for it. Do not hardcode the version anywhere else.

- **PyPI (`eq-layer`):** the `version` field of the `[project]` table in
  `pyproject.toml`. `eq_layer/_version.py` reads it through
  `importlib.metadata` when the distribution is installed, and falls back to
  reading the same field from the nearest `pyproject.toml` when running from a
  source checkout. `eq_layer.__version__` and `eq-layer doctor` report that
  value; there is no second copy in the package.
- **npm (`eq-layer`):** the `version` field in `package.json`.
  `npm/cli.mjs` reads it at runtime through `npm/version.mjs` (used by
  `--version`, `doctor`, and the help text); there is no second copy in the
  launcher.

The release workflows verify that the GitHub release tag matches these fields
before publishing, so a mismatched canonical location fails the release
instead of shipping a wrong version.

## PyPI automatic release flow

PyPI publishing is release-driven. A normal push to `main` never publishes a
package.

To publish a new version:

1. update `pyproject.toml` and `package.json` to the new version, for example
   `0.2.5`;
2. set `.release-request` to the same version and merge that change to `main`
   after the normal CI passes;
3. `.github/workflows/create-release.yml` verifies the requested version,
   creates the GitHub Release tagged `vX.Y.Z` when it is missing, and explicitly
   dispatches both publishers (`release.yml` for PyPI and `publish-npm.yml` for
   npm);
4. `.github/workflows/release.yml` verifies that the release tag and package
   version match;
5. the workflow builds the wheel/sdist, runs `twine check`, and publishes to
   PyPI through GitHub OIDC / PyPI Trusted Publishing.

The explicit npm dispatch is required because a GitHub Release created with
`GITHUB_TOKEN` does not emit a `release` event. A release tag alone does not
fan out to npm; `create-release.yml` is the component that starts both
publishers.

If the GitHub Release tag and `pyproject.toml` version differ, publishing fails
before any upload. This prevents a release tag from silently publishing the
wrong package version.

The PyPI Trusted Publisher must match:

- project: `eq-layer`
- owner: `Furox-Art`
- repository: `eq-layer`
- workflow: `release.yml`
- environment: `pypi`

No long-lived PyPI API token is required.

## Manual PyPI recovery

The same workflow has a manual dispatch input named `version`. Use it only for
recovery/retry. The supplied version must still exactly match
`pyproject.toml`.

## npm automatic release flow

The npm package uses the same semantic version as PyPI.
`.github/workflows/create-release.yml` explicitly dispatches
`publish-npm.yml` with the release version; a release tag on its own does not
publish npm. `publish-npm.yml` verifies that the requested version (release tag
or manual dispatch input) exactly matches `package.json` before uploading. It
then runs `npm test` and `npm pack --dry-run --json`.

Publishing has **two credential modes**, and OIDC is the default:

- **OIDC mode (default)** — npm trusted publishing exchanges the job's identity
  token and publishes with `npm publish --access public --provenance`, so the
  artifact carries a provenance attestation. The job declares
  `id-token: write` and installs `npm@>=11.21.0` first because older npm cannot
  request an attestation.
- **token mode (explicit only)** — a maintainer manually dispatches
  `publish-npm.yml` with `use_token_fallback=true`. That reads the `NPM_TOKEN`
  secret and publishes with `npm publish --access public --no-provenance`.
  Provenance is an OIDC-only attestation, so token mode cannot produce one.

The presence of `NPM_TOKEN` does **not** select token mode. The workflow chooses
OIDC unless the dispatch input explicitly opts into the fallback, and it fails
if the fallback is requested without a usable `NPM_TOKEN`.

### First publication of a package

The first `eq-layer@0.1.0` publication required a token because a Trusted
Publisher cannot be configured until the package already exists in the npm
registry. That historical constraint is why a granular npm access token with
**Read and write (publish and stage)**, **Bypass 2FA**, and the shortest
practical expiry may still exist as `NPM_TOKEN`. It is no longer selected
automatically.

### Trusted publishing

Trusted publishing is the default path. The npm package settings must keep a
GitHub Actions Trusted Publisher with:

1. GitHub owner/user: `Furox-Art`;
2. repository: `eq-layer`;
3. workflow filename: `publish-npm.yml`;
4. environment: leave blank;
5. direct `npm publish` permitted.

`NPM_TOKEN` is retained only for the explicit `use_token_fallback=true` recovery
path; it is not the normal release path.

## Release validation

Normal main-branch CI continues to build/check the Python distributions and
perform an npm package dry-run. Publishing a package does not change the
scientific evidence level.
