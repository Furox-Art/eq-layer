// Cross-check the canonical Python evaluation harness.
//
// This file intentionally does not reimplement affect/intent selection in JS.
// A duplicated selector can drift from the Python package and report a false
// pass. The single source of truth is eval/run.py.
//
// Usage: node tools/crosscheck.js

const { spawnSync } = require("child_process");
const path = require("path");

const root = path.join(__dirname, "..");
const candidates = process.platform === "win32"
  ? [["py", ["-3"]], ["python", []], ["python3", []]]
  : [["python3", []], ["python", []]];

let lastError = null;

for (const [exe, prefix] of candidates) {
  const result = spawnSync(
    exe,
    [...prefix, path.join(root, "eval", "run.py")],
    { cwd: root, stdio: "inherit" }
  );

  if (result.error && result.error.code === "ENOENT") {
    lastError = result.error;
    continue;
  }

  process.exit(result.status === null ? 1 : result.status);
}

console.error("No Python interpreter found; cannot run eval/run.py.");
if (lastError) console.error(lastError.message);
process.exit(2);
