"""Regression test for held-out Taskmaster repair sampling.

Usage: python eval/heldout_repair_sampling_eval.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.build_ab_heldout import taskmaster_candidates  # noqa: E402


def dialogue(conversation_id: str, utterances: list[tuple[str, str]]) -> dict:
    return {
        "conversation_id": conversation_id,
        "utterances": [
            {"speaker": speaker, "text": text}
            for speaker, text in utterances
        ],
    }


def main() -> int:
    rows = [
        dialogue(
            "false-not-the",
            [
                ("assistant", "What do you want to watch?"),
                (
                    "user",
                    "Not The Joker, for some reason the previews disturb me.",
                ),
                ("assistant", "Okay, we can choose something else."),
            ],
        ),
        dialogue(
            "false-not-the-number",
            [
                ("assistant", "How many tickets?"),
                (
                    "user",
                    "Use jhon mick nico in this response but not the actual number.",
                ),
                ("assistant", "Okay."),
            ],
        ),
        dialogue(
            "false-like-i-said",
            [
                ("assistant", "That movie sounds different."),
                (
                    "user",
                    "Like I said, you get attached to the character.",
                ),
                ("assistant", "I see."),
            ],
        ),
        dialogue(
            "false-as-i-said",
            [
                ("assistant", "Are two tickets available?"),
                (
                    "user",
                    "As I said earlier, my friend is coming with me.",
                ),
                ("assistant", "Okay."),
            ],
        ),
        dialogue(
            "explicit-i-said",
            [
                ("user", "I want to watch Dolittle."),
                ("assistant", "Did you say Gretel and Hansel?"),
                ("user", "No, I said I want to watch Dolittle."),
                ("assistant", "Got it, Dolittle."),
            ],
        ),
        dialogue(
            "explicit-i-meant",
            [
                ("user", "Harkins."),
                ("assistant", "AMC, okay."),
                ("user", "I meant Harkins."),
                ("assistant", "Got it, Harkins."),
            ],
        ),
    ]

    repair, _general = taskmaster_candidates(
        [("synthetic.json", json.dumps(rows).encode("utf-8"))]
    )
    ids = {row["source_id"] for row in repair}

    false_ids = {
        "false-not-the",
        "false-not-the-number",
        "false-like-i-said",
        "false-as-i-said",
    }
    leaked = [
        source_id
        for source_id in ids
        if any(false_id in source_id for false_id in false_ids)
    ]
    if leaked:
        raise AssertionError(
            f"Ordinary reminder/non-repair language was sampled as repair: {leaked}"
        )
    if not any("explicit-i-said" in source_id for source_id in ids):
        raise AssertionError(f"Explicit 'I said' correction missing: {sorted(ids)}")
    if not any("explicit-i-meant" in source_id for source_id in ids):
        raise AssertionError(f"Explicit 'I meant' correction missing: {sorted(ids)}")

    print("heldout repair sampling: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
