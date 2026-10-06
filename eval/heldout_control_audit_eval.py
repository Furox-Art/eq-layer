"""Engineering regression test for full held-out control-state audit.

Usage: python eval/heldout_control_audit_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from heldout_control_audit import build_control_records, summarize  # noqa: E402


class FakePipeline:
    def steer_messages(self, messages, *, annotated=None):
        text = str(messages[-1].get("content", ""))
        clarify = "clarify" in text
        metadata = {
            "intent_risk_decision": {
                "action": "clarify" if clarify else "statement",
            },
            "repair": {
                "active": "repair" in text,
            },
        }
        return messages, metadata


def main() -> int:
    cases = [
        {
            "id": "a",
            "source": {"stratum": "empathetic"},
            "transcript": [{"role": "user", "content": "context"}],
        },
        {
            "id": "b",
            "source": {"stratum": "task_repair"},
            "transcript": [{"role": "user", "content": "clarify repair"}],
        },
        {
            "id": "c",
            "source": {"stratum": "task_general"},
            "transcript": [{"role": "user", "content": "clarify"}],
        },
    ]

    records = build_control_records(cases, FakePipeline())
    if len(records) != 3:
        raise AssertionError(records)
    if records[1]["source"]["stratum"] != "task_repair":
        raise AssertionError(records[1])
    if "baseline" in records[0] or "eq" in records[0]:
        raise AssertionError("Control audit must not generate model responses.")

    report = summarize(records)
    if report["action_counts"] != {"statement": 1, "clarify": 2}:
        raise AssertionError(report)
    if report["clarification_rate_by_stratum"]["task_repair"] != 1.0:
        raise AssertionError(report)
    if report["active_repair_count_by_stratum"]["task_repair"] != 1:
        raise AssertionError(report)

    print("heldout control-state audit: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
