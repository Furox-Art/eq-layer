# Publishing EQ-Layer

EQ-Layer publishes two artifacts with the same semantic version.

- **PyPI `eq-layer`**: the Python runtime/control layer.
- **npm `eq-layer`**: the portable Agent Skill installer and CLI bridge.

Current release: **0.1.0**.

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

## npm

npm publishing is intentionally separate in
`.github/workflows/publish-npm.yml` until npm authentication/trusted publishing
is fully configured. A failed npm publication therefore cannot make a PyPI
release workflow fail.

## Release validation

Normal main-branch CI continues to build/check the Python distributions and
perform an npm package dry-run. Publishing a package does not change the
scientific evidence level.
