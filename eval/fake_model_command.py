"""Deterministic fake backend for response-experiment engineering tests."""

from __future__ import annotations

import json
import sys


def main() -> int:
    payload = json.load(sys.stdin)
    messages = payload["messages"]
    last_user = next(
        (
            str(message.get("content", ""))
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    response = {
        "text": (
            f"model={payload['model_id']} "
            f"seed={payload['seed']} "
            f"messages={len(messages)} "
            f"system={int(bool(messages and messages[0].get('role') == 'system'))} "
            f"user={last_user}"
        )
    }
    print(json.dumps(response, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
