#!/usr/bin/env python3
"""Install the canonical EQ-Layer Agent Skill into user-level agent roots."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


TARGETS = {
    "codex": Path(".codex/skills/eq-layer"),
    "claude": Path(".claude/skills/eq-layer"),
    "copilot": Path(".copilot/skills/eq-layer"),
    "cursor": Path(".cursor/skills/eq-layer"),
    "grok": Path(".grok/skills/eq-layer"),
    "common": Path(".agents/skills/eq-layer"),
}


def source_skill() -> Path:
    root = Path(__file__).resolve().parents[1]
    source = root / "skills" / "eq-layer"
    if not (source / "SKILL.md").is_file():
        raise FileNotFoundError(f"Canonical skill not found: {source}")
    return source


def install_one(source: Path, destination: Path, *, force: bool, dry_run: bool) -> str:
    if destination.exists() and not force:
        return f"skip existing: {destination}"
    if dry_run:
        return f"would install: {destination}"
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return f"installed: {destination}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        nargs="+",
        choices=(*TARGETS, "all"),
        default=["all"],
    )
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    requested = list(TARGETS) if "all" in args.target else args.target
    source = source_skill()
    seen: set[Path] = set()
    for target in requested:
        destination = (args.home / TARGETS[target]).resolve()
        if destination in seen:
            continue
        seen.add(destination)
        print(
            f"{target}: "
            + install_one(source, destination, force=args.force, dry_run=args.dry_run)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
