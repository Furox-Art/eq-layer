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

## npm automatic release flow

The npm package uses the same semantic version as PyPI. After the first npm
publication, GitHub Releases automatically publish npm as well.

For the **first** `eq-layer@0.1.0` publication, npm requires account
authorization because a Trusted Publisher cannot be configured until the
package already exists in the npm registry.

Create a granular npm access token with:

- package/scope permission: **Read and write (publish and stage)**;
- **Bypass 2FA** enabled for non-interactive CI publication;
- the shortest practical expiry.

Store it in the GitHub repository as the secret `NPM_TOKEN`. Then changing
`.npm-release-request` to the package version triggers the first publication.

After `eq-layer@0.1.0` exists on npm:

1. open the npm package settings;
2. add a GitHub Actions Trusted Publisher;
3. GitHub owner/user: `Furox-Art`;
4. repository: `eq-layer`;
5. workflow filename: `publish-npm.yml`;
6. environment: leave blank;
7. permit direct `npm publish`;
8. verify a Trusted Publisher release works;
9. remove the long-lived `NPM_TOKEN` GitHub secret.

Future GitHub Releases such as `v0.2.0` then publish to both PyPI and npm
through their release workflows. The npm workflow verifies that the GitHub
release version exactly matches `package.json` before uploading.

## Release validation

Normal main-branch CI continues to build/check the Python distributions and
perform an npm package dry-run. Publishing a package does not change the
scientific evidence level.
