"""Check the primary-endpoint ballot is built from the control, not the baseline.

The point of the calibrated control is that the ballot compares EQ against a
same-length response. Building it from the raw baseline would look identical in
code and quietly reintroduce the length confound.

These tests run the real script end to end over a fixture pairs file, because
the mapping from a response arm to the ballot slot is what has to be verified,
not the randomization helper on its own.

Usage: python eval/selected_ballot_eval.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eq_layer.ab_eval import prepare_blinded  # noqa: E402

FAILURES: list[str] = []
RAW = "RAWRAWRAWRAW"
CONTROL = "CTRLCTRLCTRL"
EQ = "EQ"


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def build_fixture(directory: str, calibration: dict, rows: list[dict]) -> tuple[str, str, str]:
    pairs_path = os.path.join(directory, "pairs.jsonl")
    with open(pairs_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    cal_path = os.path.join(directory, "cal.json")
    with open(cal_path, "w", encoding="utf-8") as handle:
        json.dump(calibration, handle)
    ballot_path = os.path.join(directory, "ballot.jsonl")
    key_path = os.path.join(directory, "key.json")
    return pairs_path, cal_path, ballot_path, key_path


def run_script(pairs: str, cal: str, ballot: str, key: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "prepare_selected_ballot.py"),
            pairs,
            cal,
            ballot,
            key,
            "--seed",
            "4242",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def fixture_rows(profile: str = "one_sentence", n: int = 4) -> list[dict]:
    return [
        {
            "id": f"c{i}",
            "source": {"stratum": "empathetic"},
            "context": [{"role": "user", "content": "hi"}],
            "baseline": f"{RAW}{i}",
            "eq": f"{EQ}{i}",
            f"length_matched_{profile}": f"{CONTROL}{i}",
        }
        for i in range(n)
    ]


CALIBRATION = {
    "selection": {
        "selected_profile": "one_sentence",
        "worst_gap": 0.0542,
        "within_tolerance": True,
        "tolerance": 0.1,
    }
}


def main() -> int:
    directory = tempfile.mkdtemp(prefix="eq-ballot-")

    # The control, not the raw baseline, has to end up in the ballot.
    pairs, cal, ballot, key = build_fixture(directory, CALIBRATION, fixture_rows())
    done = run_script(pairs, cal, ballot, key)
    check(done.returncode == 0, f"script failed: {done.stderr.strip()}")

    if done.returncode == 0:
        with open(ballot, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        with open(key, encoding="utf-8") as handle:
            decoded = json.load(handle)

        check(len(rows) == 4, f"expected 4 ballot rows, got {len(rows)}")
        for row in rows:
            text = f"{row['response_A']}{row['response_B']}"
            check(RAW not in text, f"{row['id']}: raw baseline leaked into the ballot")
            check(CONTROL in text, f"{row['id']}: control missing from the ballot")
            check(EQ in text, f"{row['id']}: EQ response missing from the ballot")
            check(
                all(v == "" for v in row["ratings"].values()),
                f"{row['id']}: ratings must start blank",
            )

        values = {v for entry in decoded.values() for v in entry.values()}
        check(
            values == {"eq", "length_matched:one_sentence"},
            f"key should name both arms honestly, got {values}",
        )
        check("baseline" not in values, "control must not be labelled baseline")

        report = json.loads(done.stdout)
        check(
            report["primary_endpoint"] == "eq vs length-matched control",
            "report must state the primary endpoint",
        )
        check(report["eq_first_slots"] + report["control_first_slots"] == 4, "slot counts must add up")

    # A profile the calibration did not select must be an error, not a silent
    # fallback to some other arm.
    missing_dir = os.path.join(directory, "m")
    os.makedirs(missing_dir, exist_ok=True)
    missing_profile = build_fixture(
        missing_dir,
        {"selection": {"selected_profile": "nope", "worst_gap": 0.5, "within_tolerance": False, "tolerance": 0.1}},
        fixture_rows(profile="one_sentence"),
    )
    bad = run_script(*missing_profile)
    check(bad.returncode != 0, "an absent selected profile must fail rather than fall back")

    # --require-match refuses to produce a ballot when nothing matched.
    unmatched_dir = os.path.join(directory, "u")
    os.makedirs(unmatched_dir, exist_ok=True)
    unmatched = build_fixture(
        unmatched_dir,
        {"selection": {"selected_profile": "one_sentence", "worst_gap": 0.4, "within_tolerance": False, "tolerance": 0.1}},
        fixture_rows(),
    )
    strict = subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "prepare_selected_ballot.py"),
            *unmatched,
            "--require-match",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    check(strict.returncode != 0, "--require-match must refuse an unmatched profile")
    check(
        "not length-controlled" in (strict.stdout + strict.stderr),
        "refusal must explain that the endpoint is not length-controlled",
    )

    # Default arm naming in the shared helper must be unchanged.
    _, default_key = prepare_blinded(
        [{"id": "a", "context": [], "baseline": "b", "eq": "e"}], seed=1
    )
    check(
        {v for entry in default_key.values() for v in entry.values()} == {"eq", "baseline"},
        "default arm names must stay eq/baseline",
    )

    print(json.dumps({"failures": FAILURES, "ok": not FAILURES}, indent=2))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())