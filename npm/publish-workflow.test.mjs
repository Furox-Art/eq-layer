import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const publishWorkflow = readFileSync(
  new URL('../.github/workflows/publish-npm.yml', import.meta.url), 'utf8'
);
const releaseWorkflow = readFileSync(
  new URL('../.github/workflows/create-release.yml', import.meta.url), 'utf8'
);

test('trusted OIDC publishing is default despite a configured legacy token', () => {
  assert.match(publishWorkflow, /use_token_fallback:[\s\S]*?default: false/);
  assert.match(publishWorkflow, /NODE_AUTH_TOKEN:\s*\$\{\{ inputs\.use_token_fallback && secrets\.NPM_TOKEN \|\| '' \}\}/);
  assert.match(publishWorkflow, /npm publish --access public --provenance/);
});

test('legacy token mode requires explicit opt-in and omits provenance', () => {
  assert.match(publishWorkflow, /if \[ "\$\{\{ inputs\.use_token_fallback \}\}" = "true" \]/);
  assert.match(publishWorkflow, /npm publish --access public --no-provenance/);
});

test('publish workflow checks out the release tag it is publishing', () => {
  // A release-triggered run must build the tagged commit, not whatever main
  // points at by the time the workflow starts.
  assert.match(
    publishWorkflow,
    /uses:\s*actions\/checkout@[\s\S]{0,400}?ref:\s*\$\{\{\s*github\.event\.release\.tag_name\b/
  );
});

test('create-release dispatches publish-npm with the verified version', () => {
  // The release workflow is the only place that knows the version it just
  // verified; omitting it would let publish-npm fall back to main.
  assert.match(
    releaseWorkflow,
    /gh workflow run publish-npm\.yml[^\n]*\bversion=\$VERSION/
  );
});
