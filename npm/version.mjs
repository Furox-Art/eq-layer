import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = dirname(here);

/**
 * Canonical npm-side version: the "version" field of package.json.
 *
 * The launcher used to carry its own hardcoded copy of the version string in
 * three places, which is exactly how launcher and package versions drift
 * apart. Read the single source instead.
 */
export function packageVersion() {
  const pkg = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
  return pkg.version;
}
