"""Static invariants for policy examples and epistemic claims.

Usage: python eval/policy_claims_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.policies import REGISTRY  # noqa: E402


STANCE_OVERCLAIMS = (
    "haklısın",
    "sen haklısın",
    "yanlış okudum",
    "yanlış okumuşum",
    "you're right",
    "you are right",
    "i was wrong",
)

SECRECY_OVERCLAIMS = (
    "kimseye söylemem",
    "kimseye anlatmam",
    "sadece bu sohbette kalır",
    "completely confidential",
    "absolute confidentiality",
    "i won't tell anyone",
)


def main() -> int:
    checked = 0

    for policy in REGISTRY.values():
        opener = policy.example_opener.lower()

        if "stance_unknown" in policy.preconditions:
            for phrase in STANCE_OVERCLAIMS:
                if phrase in opener:
                    raise AssertionError(
                        f"{policy.name} makes an unverified stance/error claim: {phrase}"
                    )
            checked += 1

        if policy.name == "boundary":
            for phrase in SECRECY_OVERCLAIMS:
                if phrase in opener:
                    raise AssertionError(
                        f"boundary promises unsupported secrecy: {phrase}"
                    )
            if "gizlilik" not in opener and "confidential" not in opener:
                raise AssertionError(
                    "boundary example should state the privacy limitation explicitly."
                )
            checked += 1

        if policy.name == "ask_one_question":
            if policy.example_opener.count("?") > 1:
                raise AssertionError("ask_one_question example contains multiple questions.")
            checked += 1

    if checked < 3:
        raise AssertionError(f"Too few policy invariants exercised: {checked}")

    print(f"policy claim invariants: {checked} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
