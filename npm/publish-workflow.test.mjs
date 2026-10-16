import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const workflow = readFileSync(
  new URL('../.github/workflows/publish-npm.yml', import.meta.url), 'utf8'
);

test('trusted OIDC publishing is default despite a configured legacy token', () => {
  assert.match(workflow, /use_token_fallback:[\s\S]*?default: false/);
  assert.match(workflow, /NODE_AUTH_TOKEN:\s*\$\{\{ inputs\.use_token_fallback && secrets\.NPM_TOKEN \|\| '' \}\}/);
  assert.match(workflow, /npm publish --access public --provenance/);
});

test('legacy token mode requires explicit opt-in and omits provenance', () => {
  assert.match(workflow, /if \[ "\$\{\{ inputs\.use_token_fallback \}\}" = "true" \]/);
  assert.match(workflow, /npm publish --access public --no-provenance/);
});
