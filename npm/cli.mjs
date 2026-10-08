#!/usr/bin/env node
import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = dirname(here);
const canonicalSkill = resolve(root, "skills", "eq-layer");

const targets = {
  codex: ".codex/skills/eq-layer",
  claude: ".claude/skills/eq-layer",
  copilot: ".copilot/skills/eq-layer",
  cursor: ".cursor/skills/eq-layer",
  grok: ".grok/skills/eq-layer",
  antigravity: ".agents/skills/eq-layer",
  common: ".agents/skills/eq-layer",
};

function die(message, code = 1) {
  console.error(message);
  process.exit(code);
}

function help() {
  console.log(`eq-layer 0.2.0

Usage:
  eq-layer install --all [--force] [--home PATH]
  eq-layer install --target codex claude copilot cursor grok antigravity [--force]
  eq-layer doctor
  eq-layer route [EQ-Layer Python route options...]

PyPI runtime:
  pip install "eq-layer[ml]"
`);
}

function parseHome(args) {
  const index = args.indexOf("--home");
  if (index === -1) return homedir();
  if (!args[index + 1]) die("--home requires a path");
  return resolve(args[index + 1]);
}

function install(args) {
  if (!existsSync(resolve(canonicalSkill, "SKILL.md"))) {
    die(`Canonical skill missing from npm package: ${canonicalSkill}`);
  }
  const force = args.includes("--force");
  const home = parseHome(args);
  let requested = [];
  if (args.includes("--all")) {
    requested = ["codex", "claude", "copilot", "cursor", "grok", "antigravity"];
  } else {
    const index = args.indexOf("--target");
    if (index === -1) die("Use --all or --target <names...>");
    for (let i = index + 1; i < args.length && !args[i].startsWith("--"); i += 1) {
      requested.push(args[i]);
    }
    if (!requested.length) die("--target requires at least one target");
  }

  const seen = new Set();
  for (const name of requested) {
    if (!(name in targets)) die(`Unknown target: ${name}`);
    const destination = resolve(home, targets[name]);
    if (seen.has(destination)) continue;
    seen.add(destination);
    if (existsSync(destination)) {
      if (!force) {
        console.log(`${name}: skip existing ${destination}`);
        continue;
      }
      rmSync(destination, { recursive: true, force: true });
    }
    mkdirSync(dirname(destination), { recursive: true });
    cpSync(canonicalSkill, destination, { recursive: true });
    console.log(`${name}: installed ${destination}`);
  }
}

function findPython() {
  for (const command of ["python", "python3"]) {
    const result = spawnSync(
      command,
      ["-c", "import eq_layer,sys; print(sys.executable)"],
      { encoding: "utf8" },
    );
    if (result.status === 0) return { command, executable: result.stdout.trim() };
  }
  return null;
}

function doctor() {
  const python = findPython();
  console.log(JSON.stringify({
    npmPackage: "eq-layer",
    version: "0.2.0",
    skillPresent: existsSync(resolve(canonicalSkill, "SKILL.md")),
    pythonRuntime: python,
    recommendation: python
      ? "EQ-Layer Python runtime is available."
      : 'Install the runtime with: pip install "eq-layer[ml]"',
  }, null, 2));
}

function route(args) {
  const python = findPython();
  if (!python) {
    die('EQ-Layer Python runtime not found. Install: pip install "eq-layer[ml]"');
  }
  const result = spawnSync(
    python.command,
    ["-m", "eq_layer.cli", "route", ...args],
    { stdio: "inherit" },
  );
  process.exit(result.status ?? 1);
}

const args = process.argv.slice(2);
const command = args.shift();
if (!command || command === "--help" || command === "-h") {
  help();
} else if (command === "--version" || command === "-v") {
  console.log("0.2.0");
} else if (command === "install") {
  install(args);
} else if (command === "doctor") {
  doctor();
} else if (command === "route") {
  route(args);
} else {
  die(`Unknown command: ${command}`);
}
