from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass

COARSE_DISCOURSE_URL = (
    "https://zissou.infosci.cornell.edu/convokit/datasets/"
    "reddit-coarse-discourse-corpus/reddit-coarse-discourse-corpus.zip"
)


@dataclass(frozen=True)
class CoarseDiscourseExample:
    text: str
    label: str
    conversation_id: str
    reply_to: str | None


def download_archive() -> bytes:
    request = urllib.request.Request(
        COARSE_DISCOURSE_URL,
        headers={"User-Agent": "eq-layer-coarse-discourse/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _find_member(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {suffix} member, found {matches}")
    return matches[0]


def load_examples(data: bytes) -> tuple[list[CoarseDiscourseExample], list[str]]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        utterances_member = _find_member(names, "utterances.jsonl")
        examples: list[CoarseDiscourseExample] = []

        with archive.open(utterances_member) as fh:
            for raw in fh:
                row = json.loads(raw.decode("utf-8"))
                text = str(row.get("text") or "").strip()
                meta = row.get("meta") or {}
                label = str(meta.get("majority_type") or "").strip().lower()
                if not text or not label:
                    continue
                examples.append(
                    CoarseDiscourseExample(
                        text=text,
                        label=label,
                        conversation_id=str(row.get("conversation_id") or ""),
                        reply_to=(
                            str(row.get("reply_to"))
                            if row.get("reply_to") is not None
                            else None
                        ),
                    )
                )
        return examples, names


def inventory(examples: list[CoarseDiscourseExample]) -> dict:
    counts = Counter(example.label for example in examples)
    return {
        "examples": len(examples),
        "conversations": len({x.conversation_id for x in examples if x.conversation_id}),
        "labels": dict(sorted(counts.items())),
    }
