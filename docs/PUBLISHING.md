# Publishing EQ-Layer

EQ-Layer publishes two artifacts with the same semantic version.

- **PyPI `eq-layer`**: the Python runtime/control layer.
- **npm `eq-layer`**: the portable Agent Skill installer and CLI bridge.

Current release: **0.1.0**.

## Release validation

Every main-branch CI run builds and checks the Python distributions and performs
an npm package dry-run. A registry publish must not bypass these checks.

## PyPI: one-time trusted-publisher setup

PyPI supports creating a new project through a **pending Trusted Publisher**.
Before the first publish, configure:

- PyPI project name: `eq-layer`
- GitHub owner: `Furox-Art`
- GitHub repository: `eq-layer`
- Workflow: `release.yml`
- Environment: leave blank unless you intentionally add one later

The release workflow uses GitHub OIDC and does not require a long-lived PyPI
token.

## npm: first publish and trusted publishing

npm trusted publishing is configured from an existing package's settings. For
the first publication, add a repository secret named `NPM_TOKEN` with
permission to publish `eq-layer`. The release workflow provides that token as a
fallback.

After the first npm release:

1. Open the `eq-layer` package settings on npm.
2. Add GitHub Actions as a Trusted Publisher for `Furox-Art/eq-layer` and
   `.github/workflows/release.yml`.
3. Allow direct `npm publish`.
4. Remove the long-lived `NPM_TOKEN` repository secret.

npm 11.5.1+ automatically prefers OIDC in a supported GitHub Actions
environment, falling back to the token only when needed.

## Triggering a release

The workflow can be run manually. The repository also contains
`.release-trigger`; changing it to the next version triggers the release
workflow from `main`.

Before changing the trigger:

1. update the version in `pyproject.toml`, `package.json`, and CLI metadata;
2. run CI;
3. ensure the corresponding registry publisher/authentication is configured;
4. change `.release-trigger` as the final release commit.

## Claim boundary

Publishing a package does not change the scientific evidence level. The
120-case generation is complete and blinded-ballot ready, but human preference
adjudication and independent loss-matrix calibration remain separate empirical
steps.
