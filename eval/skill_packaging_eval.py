"""Validate portable EQ-Layer Agent Skill packaging and local routing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SKILLS = (
    "skills/eq-layer/SKILL.md",
    ".agents/skills/eq-layer/SKILL.md",
    ".codex/skills/eq-layer/SKILL.md",
    ".claude/skills/eq-layer/SKILL.md",
    ".github/skills/eq-layer/SKILL.md",
    ".cursor/skills/eq-layer/SKILL.md",
    ".grok/skills/eq-layer/SKILL.md",
)


def main() -> int:
    for relative in REQUIRED_SKILLS:
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"missing skill: {relative}")
        if "name: eq-layer" not in path.read_text(encoding="utf-8"):
            raise AssertionError(f"invalid skill frontmatter: {relative}")

    manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
    if manifest.get("$schema") != "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json":
        raise AssertionError("portable plugin schema mismatch")
    if manifest.get("name") != "eq-layer":
        raise AssertionError("portable plugin name mismatch")

    payload = [{"role": "user", "content": "devam et"}]
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "skills/eq-layer/scripts/route.py"),
            "--intent-mode",
            "heuristic",
        ],
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
        check=False,
        cwd=ROOT,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    routed = json.loads(completed.stdout)
    if routed["runtime_mode"] != "lightweight-local":
        raise AssertionError(routed["runtime_mode"])
    if not routed["reference"]["active"] or routed["reference"]["resolved"]:
        raise AssertionError(routed["reference"])
    action = routed["factored_action"]
    if action["task_move"] != "clarify_reference":
        raise AssertionError(action)
    if action["realization"]["question_budget"] != 1:
        raise AssertionError(action)
    if not action["realization"]["no_guess"]:
        raise AssertionError(action)

    install = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/install_agent_skills.py"),
            "--target",
            "all",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=ROOT,
    )
    if install.returncode != 0:
        raise AssertionError(install.stderr)
    for target in ("codex", "claude", "copilot", "cursor", "grok", "common"):
        if f"{target}:" not in install.stdout:
            raise AssertionError(f"installer omitted {target}")

    print("agent skill packaging: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
