"""Build a deterministic external held-out A/B case set.

Sources are downloaded at runtime and are not vendored into EQ-Layer.

Composition:
- 60 EmpatheticDialogues test conversations (emotional/open-domain)
- 30 Taskmaster-3 repair/clarification user turns
- 30 Taskmaster-3 general task user turns

The output contains only conversation prefixes ending in a user turn. Human
reference replies are never included in the A/B model input.

Usage:
  python tools/build_ab_heldout.py --output heldout.jsonl --manifest heldout.manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

EMPATHETIC_URL = (
    "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/"
    "empatheticdialogues.tar.gz"
)
EMPATHETIC_REPO_COMMIT = "9649114c71e1af32189a3973b3598dc311297560"

TASKMASTER_COMMIT = "d92cb6af3005f1dc09c39e75e7daf4a04905e00b"
TASKMASTER_FILES = (
    "TM-3-2020/data/data_00.json",
    "TM-3-2020/data/data_01.json",
)
TASKMASTER_URLS = tuple(
    "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/"
    f"{TASKMASTER_COMMIT}/{path}"
    for path in TASKMASTER_FILES
)

REPAIR_RE = re.compile(
    r"\b("
    r"no[,. ]|"
    r"i said|"
    r"i meant|"
    r"meant to say|"
    r"not that|"
    r"not the|"
    r"actually|"
    r"that's wrong|"
    r"that is wrong|"
    r"you mean|"
    r"didn't say|"
    r"did not say|"
    r"sorry[, ]+i"
    r")\b",
    re.IGNORECASE,
)


def download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "eq-layer-heldout-builder/0.1"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalise_text(text: str) -> str:
    # EmpatheticDialogues uses _comma_ escapes in the released CSV.
    return " ".join(
        text.replace("_comma_", ",")
        .replace("_pipe_", "|")
        .split()
    ).strip()


def _speaker_roles(rows: list[dict]) -> dict[str, str]:
    seen: list[str] = []
    for row in rows:
        speaker = str(row.get("speaker_idx") or "")
        if speaker and speaker not in seen:
            seen.append(speaker)
    if not seen:
        return {}
    mapping = {seen[0]: "user"}
    if len(seen) > 1:
        mapping[seen[1]] = "assistant"
    for speaker in seen[2:]:
        mapping[speaker] = "assistant"
    return mapping


def empathetic_candidates(archive_bytes: bytes) -> list[dict]:
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.lower().endswith("test.csv")
        ]
        if len(members) != 1:
            raise RuntimeError(
                f"Expected one EmpatheticDialogues test.csv, found {[m.name for m in members]}"
            )
        extracted = archive.extractfile(members[0])
        if extracted is None:
            raise RuntimeError("Could not read EmpatheticDialogues test.csv")
        text = io.TextIOWrapper(extracted, encoding="utf-8")
        rows = list(csv.DictReader(text))

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        conv_id = str(row.get("conv_id") or row.get("conversation_id") or "").strip()
        utterance = _normalise_text(str(row.get("utterance") or ""))
        if conv_id and utterance:
            grouped[conv_id].append(row)

    candidates: list[dict] = []
    for conv_id, turns in grouped.items():
        turns.sort(key=lambda row: int(row.get("utterance_idx") or 0))
        role_map = _speaker_roles(turns)
        messages = [
            {
                "role": role_map.get(str(row.get("speaker_idx") or ""), "assistant"),
                "content": _normalise_text(str(row.get("utterance") or "")),
            }
            for row in turns
        ]

        # Pick the latest user turn that has a human assistant response after it.
        chosen_index = None
        for idx in range(len(messages) - 1):
            if messages[idx]["role"] == "user" and messages[idx + 1]["role"] == "assistant":
                chosen_index = idx
        if chosen_index is None:
            continue

        prefix = messages[max(0, chosen_index - 5): chosen_index + 1]
        if not prefix or prefix[-1]["role"] != "user":
            continue

        candidates.append(
            {
                "source_id": conv_id,
                "transcript": prefix,
                "stratum": "empathetic",
            }
        )

    return candidates


def _tm_messages(dialogue: dict) -> list[dict]:
    messages = []
    for turn in dialogue.get("utterances") or []:
        speaker = str(turn.get("speaker") or "").lower()
        if speaker not in {"user", "assistant"}:
            continue
        text = _normalise_text(str(turn.get("text") or ""))
        if text:
            messages.append({"role": speaker, "content": text})
    return messages


def taskmaster_candidates(blobs: list[tuple[str, bytes]]) -> tuple[list[dict], list[dict]]:
    repair: list[dict] = []
    general: list[dict] = []

    for source_name, data in blobs:
        dialogues = json.loads(data.decode("utf-8"))
        if not isinstance(dialogues, list):
            raise RuntimeError(f"Taskmaster file is not a list: {source_name}")

        for dialogue in dialogues:
            conv_id = str(
                dialogue.get("conversation_id")
                or dialogue.get("conversationId")
                or ""
            ).strip()
            messages = _tm_messages(dialogue)
            if not conv_id or len(messages) < 2:
                continue

            user_indices = [
                idx
                for idx, message in enumerate(messages[:-1])
                if message["role"] == "user"
                and messages[idx + 1]["role"] == "assistant"
            ]
            if not user_indices:
                continue

            repair_indices = [
                idx
                for idx in user_indices
                if idx > 0 and REPAIR_RE.search(messages[idx]["content"])
            ]
            chosen_repair = repair_indices[-1] if repair_indices else None
            chosen_general = next(
                (
                    idx
                    for idx in reversed(user_indices)
                    if idx != chosen_repair
                    and not REPAIR_RE.search(messages[idx]["content"])
                ),
                None,
            )

            if chosen_repair is not None:
                repair.append(
                    {
                        "source_id": f"{source_name}:{conv_id}:{chosen_repair}",
                        "transcript": messages[
                            max(0, chosen_repair - 5): chosen_repair + 1
                        ],
                        "stratum": "task_repair",
                    }
                )
            if chosen_general is not None:
                general.append(
                    {
                        "source_id": f"{source_name}:{conv_id}:{chosen_general}",
                        "transcript": messages[
                            max(0, chosen_general - 5): chosen_general + 1
                        ],
                        "stratum": "task_general",
                    }
                )

    return repair, general


def deterministic_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    if len(rows) < n:
        raise RuntimeError(f"Need {n} cases, found only {len(rows)}")
    rng = random.Random(seed)
    picked = rng.sample(rows, n)
    return sorted(picked, key=lambda row: row["source_id"])


def build(seed: int = 20261005) -> tuple[list[dict], dict]:
    empathetic_archive = download(EMPATHETIC_URL)
    tm_blobs = [
        (path, download(url))
        for path, url in zip(TASKMASTER_FILES, TASKMASTER_URLS, strict=True)
    ]

    empathetic = empathetic_candidates(empathetic_archive)
    repair, general = taskmaster_candidates(tm_blobs)

    selected = (
        deterministic_sample(empathetic, 60, seed + 1)
        + deterministic_sample(repair, 30, seed + 2)
        + deterministic_sample(general, 30, seed + 3)
    )

    output = []
    for index, row in enumerate(selected, start=1):
        output.append(
            {
                "id": f"external-{index:03d}",
                "transcript": row["transcript"],
                "source": {
                    "corpus": (
                        "EmpatheticDialogues"
                        if row["stratum"] == "empathetic"
                        else "Taskmaster-3"
                    ),
                    "source_id": row["source_id"],
                    "stratum": row["stratum"],
                },
            }
        )

    manifest = {
        "design": "external-heldout-mixed-dialogue-prefixes",
        "seed": seed,
        "n_cases": len(output),
        "strata": {
            "empathetic": 60,
            "task_repair": 30,
            "task_general": 30,
        },
        "sources": {
            "EmpatheticDialogues": {
                "url": EMPATHETIC_URL,
                "repository_commit": EMPATHETIC_REPO_COMMIT,
                "license": "CC BY-NC 4.0",
                "archive_sha256": sha256(empathetic_archive),
                "candidate_count": len(empathetic),
            },
            "Taskmaster-3": {
                "repository_commit": TASKMASTER_COMMIT,
                "license": "CC BY 4.0",
                "files": [
                    {
                        "path": path,
                        "url": url,
                        "sha256": sha256(data),
                    }
                    for (path, data), url in zip(
                        tm_blobs,
                        TASKMASTER_URLS,
                        strict=True,
                    )
                ],
                "repair_candidate_count": len(repair),
                "general_candidate_count": len(general),
            },
        },
        "selection_rule": (
            "Deterministic sampling from external corpora; each case is a "
            "conversation prefix ending in a user turn with the human reference "
            "reply excluded from model input."
        ),
        "training_overlap_policy": (
            "These corpora are distinct from EQ-Layer's EmoBank, XDailyDialog, "
            "Coarse Discourse, DialogBank, and DBDC3 development/training sources."
        ),
    }
    return output, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()

    cases, manifest = build(seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in cases:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    case_bytes = output_path.read_bytes()
    manifest["heldout_sha256"] = sha256(case_bytes)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
