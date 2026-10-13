# Publishing EQ-Layer

EQ-Layer publishes two artifacts with the same semantic version.

- **PyPI `eq-layer`**: the Python runtime/control layer.
- **npm `eq-layer`**: the portable Agent Skill installer and CLI bridge.

Current release: **0.2.2**.

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

1. update `pyproject.toml` to the new version, for example `0.2.0`;
2. let the normal CI pass on `main`;
3. create and publish a GitHub Release tagged `v0.2.0`;
4. `.github/workflows/release.yml` verifies that the release tag and package
   version match;
5. the workflow builds the wheel/sdist, runs `twine check`, and publishes to
   PyPI through GitHub OIDC / PyPI Trusted Publishing.

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

The npm package uses the same semantic version as PyPI. A GitHub Release tagged
`vX.Y.Z` publishes npm as well, and `.github/workflows/publish-npm.yml` verifies
that the requested version (release tag, manual dispatch input, or
`.npm-release-request`) exactly matches `package.json` before uploading. It then
runs `npm test` and `npm pack --dry-run --json`.

Publishing has **two credential modes, selected at run time** by whether the
`NPM_TOKEN` secret is configured:

- **token mode** - `NPM_TOKEN` is set: the workflow publishes through the
  classic `NODE_AUTH_TOKEN` credential with
  `npm publish --access public --no-provenance`;
- **OIDC mode** - no `NPM_TOKEN` is set: npm trusted publishing exchanges the
  job's identity token and publishes with
  `npm publish --access public --provenance`.

Provenance is an OIDC-only attestation, so the `--provenance` flag is passed in
OIDC mode only; with a token configured it would fail with "cannot be used with
a token". The workflow installs `npm@>=11.21.0` first because older npm cannot
request an attestation at all. The mode is a repository-settings decision, not a
workflow edit: add or remove the secret and the next release switches.

### First publication of a package

The first `eq-layer@0.1.0` publication required a token because a Trusted
Publisher cannot be configured until the package already exists in the npm
registry. Create a granular npm access token with:

- package/scope permission: **Read and write (publish and stage)**;
- **Bypass 2FA** enabled for non-interactive CI publication;
- the shortest practical expiry.

Store it in the GitHub repository as the secret `NPM_TOKEN`. While that secret is
present, every npm release publishes in token mode.

### Moving to trusted publishing (optional)

Once the package exists on npm:

1. open the npm package settings;
2. add a GitHub Actions Trusted Publisher;
3. GitHub owner/user: `Furox-Art`;
4. repository: `eq-layer`;
5. workflow filename: `publish-npm.yml`;
6. environment: leave blank;
7. permit direct `npm publish`;
8. verify a trusted-publishing release works;
9. remove the long-lived `NPM_TOKEN` GitHub secret.

Removing the secret is what switches the workflow to OIDC mode, so step 9 is the
cutover. Until then token mode keeps publishing without provenance attestation.

## Release validation

Normal main-branch CI continues to build/check the Python distributions and
perform an npm package dry-run. Publishing a package does not change the
scientific evidence level.
