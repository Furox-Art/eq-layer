"""Measure whether the model backend is byte-reproducible across runs.

At temperature 0 with a fixed per-case seed, generation is only deterministic
given identical logits, and CPU matmul reduction order varies between runs.
Two pilot runs produced identical text for 8 of 12 EQ arms and 7 of 12 baseline
arms, so the pairing rule in the manifest overstated what was actually held
constant.

This sends the same payload twice to one server process, then to a fresh
process, and reports how much text matched. Within-process repetition isolates
leaked state between requests; across-process comparison isolates kernel-level
variance.

Usage:
  python eval/reproducibility_check.py \
    --model-command "python eval/hf_model_server.py --model-id Qwen/Qwen2.5-0.5B-Instruct"
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

PAYLOAD = {
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "seed": 42,
    "temperature": 0.0,
    "messages": [
        {"role": "user", "content": "Randevumu üç kez değiştirdiler"},
        {"role": "assistant", "content": "Anlıyorum"},
        {"role": "user", "content": "Ne yapacağım?"},
    ],
}


def request(command: list[str], payload: dict) -> str:
    completed = subprocess.run(
        command,
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
        timeout=900,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"model command failed: {completed.stderr.strip()}")
    return json.loads(completed.stdout.strip().splitlines()[-1])["text"]


def persistent_requests(command: list[str], payload: dict, repeats: int) -> list[str]:
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None and process.stdout is not None
    out = []
    try:
        for _ in range(repeats):
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
            line = process.stdout.readline().strip()
            out.append(json.loads(line)["text"])
    finally:
        process.stdin.close()
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-command", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()

    command = shlex.split(args.model_command)

    within = persistent_requests(command, PAYLOAD, args.repeats)
    across = [request(command, PAYLOAD) for _ in range(2)]

    within_same = sum(1 for t in within if t == within[0])
    across_same = sum(1 for t in across if t == across[0])
    cross_match = sum(1 for t in within if t == across[0])

    report = {
        "audit": "generation-reproducibility",
        "model_id": PAYLOAD["model_id"],
        "temperature": PAYLOAD["temperature"],
        "seed": PAYLOAD["seed"],
        "within_process": {
            "repeats": args.repeats,
            "identical_to_first": within_same,
            "reproducible": within_same == args.repeats,
        },
        "across_process": {
            "repeats": 2,
            "identical_to_first": across_same,
            "reproducible": across_same == 2,
        },
        "within_matches_across": cross_match,
        "texts": {"within": within, "across": across},
        "interpretation": (
            "within_process=false means request state leaks between calls in the "
            "persistent backend. across_process=false means the kernels are "
            "non-deterministic and a paired design cannot rely on identical text "
            "for identical seeds."
        ),
        "claim_boundary": (
            "This checks one payload on one machine. It does not certify "
            "reproducibility for the full held-out set or across runner types."
        ),
    }

    text = json.dumps(report, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())