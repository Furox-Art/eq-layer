from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass

DBDC3_URL = "https://dbd-challenge.github.io/dbdc3/data/DBDC3.zip"
DBDC3_ARCHIVE_SHA256 = "736229795dc3732f8e6bb421f094dc820ef944fef9d9d320d4110ef992b60e85"


@dataclass(frozen=True)
class BreakdownTurn:
    source_file: str
    split: str
    dialogue_id: str
    turn_index: int
    text: str
    context: tuple[str, ...]
    counts: dict[str, int]

    @property
    def model_text(self) -> str:
        context = "\n".join(self.context)
        if context:
            return f"{context}\n[SYSTEM]\n{self.text}"
        return self.text

    @property
    def score(self) -> float:
        total = sum(self.counts.values())
        if not total:
            return 0.0
        weighted = (
            self.counts.get("T", 0) * 0.5
            + self.counts.get("X", 0) * 1.0
        )
        return weighted / total

    @property
    def hard_breakdown(self) -> bool:
        total = sum(self.counts.values())
        if not total:
            return False
        return self.counts.get("X", 0) / total >= 0.5


def download_archive() -> bytes:
    request = urllib.request.Request(
        DBDC3_URL,
        headers={"User-Agent": "eq-layer-dbdc3-audit/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _iter_json_members(archive: zipfile.ZipFile) -> list[str]:
    return sorted(
        name
        for name in archive.namelist()
        if name.lower().endswith(".json")
    )


def _english_member(name: str) -> bool:
    low = name.lower().replace("\\", "/")
    return "/en/" in f"/{low}"


def _effective_english_members(names: list[str]) -> list[str]:
    english = [name for name in names if _english_member(name)]
    revised = [
        name
        for name in english
        if "/dbdc3_revised/en/" in name.lower().replace("\\", "/")
    ]
    if revised:
        return revised
    return [
        name
        for name in english
        if "/dbdc3/en/" in name.lower().replace("\\", "/")
    ]


def _split_from_member(name: str) -> str:
    low = name.lower().replace("\\", "/")
    if "/en/dev/" in low:
        return "dev"
    if "/en/eval/" in low or "/en/test/" in low:
        return "eval"
    return "unknown"


def _dialogue_id(payload: dict, member: str) -> str:
    for key in ("dialogue-id", "dialogue_id", "dialog-id", "id"):
        if payload.get(key) is not None:
            return str(payload[key])
    return member.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def parse_english_turns(data: bytes) -> tuple[list[BreakdownTurn], dict]:
    turns: list[BreakdownTurn] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = _iter_json_members(archive)
        all_english = [name for name in members if _english_member(name)]
        english = _effective_english_members(members)

        for member in english:
            with archive.open(member) as fh:
                payload = json.load(fh)

            dialogues = payload if isinstance(payload, list) else [payload]
            for dialogue in dialogues:
                if not isinstance(dialogue, dict):
                    continue
                dialogue_turns = dialogue.get("turns")
                if not isinstance(dialogue_turns, list):
                    continue

                history: list[str] = []
                did = _dialogue_id(dialogue, member)
                for position, turn in enumerate(dialogue_turns):
                    if not isinstance(turn, dict):
                        continue
                    text = str(turn.get("utterance") or "").strip()
                    speaker = str(turn.get("speaker") or "").upper()
                    annotations = turn.get("annotations") or []

                    if speaker == "S" and text:
                        counts = Counter()
                        for ann in annotations:
                            if not isinstance(ann, dict):
                                continue
                            label = str(ann.get("breakdown") or "").upper()
                            if label in {"O", "T", "X"}:
                                counts[label] += 1
                        if counts:
                            turns.append(
                                BreakdownTurn(
                                    source_file=member,
                                    split=_split_from_member(member),
                                    dialogue_id=did,
                                    turn_index=int(turn.get("turn-index", position)),
                                    text=text,
                                    context=tuple(history[-3:]),
                                    counts=dict(counts),
                                )
                            )

                    if text:
                        history.append(f"{speaker}: {text}")

        meta = {
            "zip_members": len(members),
            "all_english_json_members": len(all_english),
            "effective_english_json_members": len(english),
            "effective_source": (
                "dbdc3_revised"
                if any("/dbdc3_revised/en/" in name.lower().replace("\\", "/") for name in english)
                else "dbdc3"
            ),
            "effective_dev_members": sum("/en/dev/" in name.lower().replace("\\", "/") for name in english),
            "effective_eval_members": sum(
                "/en/eval/" in name.lower().replace("\\", "/")
                or "/en/test/" in name.lower().replace("\\", "/")
                for name in english
            ),
        }
    return turns, meta


def inventory(turns: list[BreakdownTurn]) -> dict:
    aggregate = Counter()
    hard = 0
    for turn in turns:
        aggregate.update(turn.counts)
        hard += int(turn.hard_breakdown)

    return {
        "system_turns": len(turns),
        "dialogues": len({turn.dialogue_id for turn in turns}),
        "annotation_counts": dict(sorted(aggregate.items())),
        "hard_breakdown_turns": hard,
        "hard_breakdown_rate": round(hard / len(turns), 4) if turns else 0.0,
        "mean_breakdown_score": round(
            sum(turn.score for turn in turns) / len(turns),
            4,
        ) if turns else 0.0,
    }
