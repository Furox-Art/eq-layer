"""Check that held-out sampling margins are reported and correctly flagged.

The builder samples a stratum out of a candidate pool. If a pool shrinks close
to its quota the build still succeeds while real diversity drops, so the
margin is recorded in the manifest and warned about on stderr. This test pins
that behaviour, including the current known-narrow task_repair pool.

Usage: python eval/heldout_sampling_margin_eval.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.build_ab_heldout import (  # noqa: E402
    MIN_POOL_HEADROOM_FACTOR,
    narrow_strata,
    sampling_margins,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def main() -> int:
    wide = sampling_margins(
        {
            "empathetic": (2540, 60),
            "task_general": (2376, 30),
        }
    )
    check(wide["empathetic"]["headroom"] == 2480, "headroom should be pool minus drawn")
    check(wide["empathetic"]["draw_ratio"] == 0.0236, "draw_ratio should be drawn/pool")
    check(wide["empathetic"]["narrow_pool"] is False, "2540 into 60 is not narrow")
    check(narrow_strata(wide) == [], "no narrow strata expected for wide pools")

    # Boundary: exactly at the factor is not narrow, below it is.
    at_factor = sampling_margins({"x": (60, 30)})  # 30 * 2.0 == 60
    check(at_factor["x"]["narrow_pool"] is False, "pool exactly at factor is not narrow")
    below = sampling_margins({"x": (59, 30)})
    check(below["x"]["narrow_pool"] is True, "pool below factor must be narrow")
    check(narrow_strata(below) == ["x"], "narrow_strata should list the offending stratum")

    # The real task_repair pool, recorded from the 2026-10-06 frozen build.
    repair = sampling_margins({"task_repair": (58, 30)})
    check(
        repair["task_repair"]["narrow_pool"] is True,
        "task_repair at 58 candidates into 30 must be flagged narrow",
    )
    check(
        repair["task_repair"]["headroom"] == 28,
        "task_repair headroom should be 28",
    )
    check(
        58 < 30 * MIN_POOL_HEADROOM_FACTOR,
        "fixture no longer exercises the threshold; update the recorded pool size",
    )

    # Empty pool must not divide by zero.
    empty = sampling_margins({"empty": (0, 30)})
    check(empty["empty"]["draw_ratio"] is None, "empty pool draw_ratio must be None")
    check(empty["empty"]["narrow_pool"] is True, "empty pool must be flagged narrow")

    print(
        json.dumps(
            {
                "min_pool_headroom_factor": MIN_POOL_HEADROOM_FACTOR,
                "narrow_strata_now": narrow_strata(
                    sampling_margins(
                        {"empathetic": (2540, 60), "task_repair": (58, 30), "task_general": (2376, 30)}
                    )
                ),
                "failures": FAILURES,
                "ok": not FAILURES,
            },
            indent=2,
        )
    )
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())