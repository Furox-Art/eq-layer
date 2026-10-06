#!/usr/bin/env python3
"""Skill-local wrapper around the installed EQ-Layer Python CLI."""

from __future__ import annotations

import sys
from pathlib import Path


def _make_repo_importable() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "eq_layer").is_dir():
            sys.path.insert(0, str(parent))
            return


_make_repo_importable()

from eq_layer.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["route", *sys.argv[1:]]))
