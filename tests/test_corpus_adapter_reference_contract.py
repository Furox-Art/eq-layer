"""Offline reference fixtures for the three previously untested corpus readers.

All fixtures are generated in memory.  No third-party download or change to
the actual corpora is required to validate parsing, metadata and provenance.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from unittest.mock import Mock

import pytest

from eq_layer import coarse_discourse, dbdc3, dialogbank_iso


def _zip(**members: object) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as z:
        for name, value in members.items():
            text = value if isinstance(value, str) else json.dumps(value)
            z.writestr(name, text)
    return output.getvalue()


def test_coarse_discourse_extracts_reply_context_and_inventory() -> None:
    rows = [
        {
            "id": "root",
            "text": "First turn",
            "conversation_id": "c1",
            "meta": {"majority_type": "Question"},
        },
        {
            "id": "reply",
            "text": "I disagree",
            "conversation_id": "c1",
            "reply_to": "root",
            "meta": {"majority_type": "Answer"},
        },
        {"id": "blank", "text": "", "conversation_id": "c2", "meta": {"majority_type": "Answer"}},
        {"id": "unlabelled", "text": "Hello", "conversation_id": "c3", "meta": {}},
    ]
    archive = _zip(**{"corpus/utterances.jsonl": "\n".join(json.dumps(row) for row in rows)})
    examples, members = coarse_discourse.load_examples(archive)
    assert len(examples) == 2
    assert members == ["corpus/utterances.jsonl"]
    assert examples[0].model_text == "First turn"
    assert examples[1].model_text == "First turn\n[REPLY]\nI disagree"
    assert examples[1].reply_to == "root"
    assert coarse_discourse.inventory(examples) == {
        "examples": 2,
        "conversations": 1,
        "labels": {"answer": 1, "question": 1},
    }
    assert coarse_discourse.sha256_bytes(archive) == hashlib.sha256(archive).hexdigest()


def test_coarse_discourse_rejects_missing_and_ambiguous_members() -> None:
    for archive in (
        _zip(**{"other.jsonl": ""}),
        _zip(**{"a/utterances.jsonl": "", "b/utterances.jsonl": ""}),
    ):
        with pytest.raises(ValueError, match="Expected one utterances.jsonl"):
            coarse_discourse.load_examples(archive)


def test_dbdc3_revised_english_takes_precedence_and_retains_annotation_counts() -> None:
    original = {
        "dialogue-id": "outdated",
        "turns": [{"speaker": "S", "utterance": "obsolete", "annotations": [{"breakdown": "X"}]}],
    }
    revised = [
        {
            "dialogue-id": "correct",
            "turns": [
                {"speaker": "U", "utterance": "Could you help?"},
                {
                    "speaker": "S",
                    "utterance": "Of course",
                    "turn-index": 17,
                    "annotations": [
                        {"breakdown": "O"},
                        {"breakdown": "X"},
                        {"breakdown": "X"},
                        {"breakdown": "?"},
                    ],
                },
                {"speaker": "S", "utterance": "Unrated", "annotations": []},
                {"speaker": "U", "utterance": "Great"},
            ],
        }
    ]
    data = _zip(
        **{
            "archive/DBDC3/en/dev/dialogue.json": original,
            "archive/DBDC3_Revised/en/eval/dialogue.json": revised,
            "archive/DBDC3_Revised/ja/eval/dialogue.json": revised,
            "readme.txt": "notes",
        }
    )
    turns, metadata = dbdc3.parse_english_turns(data)
    assert len(turns) == 1
    turn = turns[0]
    assert turn.dialogue_id == "correct"
    assert turn.turn_index == 17
    assert turn.context == ("U: Could you help?",)
    assert turn.model_text == "U: Could you help?\n[SYSTEM]\nOf course"
    assert turn.counts == {"O": 1, "X": 2}
    assert turn.hard_breakdown
    assert turn.score == pytest.approx(2 / 3)
    assert metadata["effective_source"] == "dbdc3_revised"
    assert metadata["all_english_json_members"] == 2
    assert dbdc3.inventory(turns)["hard_breakdown_turns"] == 1
    assert dbdc3.sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_dbdc3_fallback_and_unrated_cases() -> None:
    data = _zip(
        **{
            "archive/DBDC3/en/dev/simple.json": {
                "turns": [
                    "not an object",
                    {"speaker": "U", "utterance": "Thanks"},
                    {"speaker": "S", "utterance": "Ok", "annotations": [{"breakdown": "T"}]},
                ],
            },
            "archive/DBDC3/en/eval/broken.json": {"turns": "not a list"},
        }
    )
    turns, metadata = dbdc3.parse_english_turns(data)
    assert len(turns) == 1
    assert turns[0].score == pytest.approx(0.5)
    assert not turns[0].hard_breakdown
    assert metadata["effective_source"] == "dbdc3"
    assert metadata["effective_dev_members"] == 1
    assert metadata["effective_eval_members"] == 1
    assert dbdc3.inventory([])["mean_breakdown_score"] == 0.0
    assert dbdc3.inventory([])["hard_breakdown_rate"] == 0.0


XML = """<root>
  <w xml:id="w1">Please</w><w xml:id="w2">continue</w>
  <spanGrp type="functionalVerbalSegment" xml:id="s1">
    <span from="#w1" to="#w2"/>
  </spanGrp>
  <fs type="functionalSegment" xml:id="f1">
    <f name="verbalComponent" fVal="#s1"/>
  </fs>
  <dialogueAct target="#f1" dimension="Task" communicativeFunction="Request"/>
</root>"""


def test_dialogbank_structured_xml_produces_real_annotation_text() -> None:
    examples = dialogbank_iso.parse_diaml(XML, "memory://sample.diaml")
    assert examples == [
        dialogbank_iso.ISOExample("memory://sample.diaml", "Please continue", "Request", "Task")
    ]
    assert dialogbank_iso.inventory(examples) == {
        "dialogues": 1,
        "examples": 1,
        "functions": {"Request": 1},
    }


def test_dialogbank_repair_and_legacy_fallback_are_exercised() -> None:
    repaired = XML.replace("Please", "Please & carry on")
    assert "Please & carry on" in dialogbank_iso.parse_diaml(repaired, "repair")[0].text
    malformed = XML + "<not-closed"
    recovered = dialogbank_iso.parse_diaml(malformed, "legacy")
    assert len(recovered) == 1
    assert recovered[0].text == "Please continue"
    with pytest.raises(ValueError, match="recovered no dialogue acts"):
        dialogbank_iso.parse_diaml("<broken>", "invalid")


def test_dialogbank_discovers_links_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = iter(
        [
            '<a href="/data/dialogue.diaml">A</a><a href="/notes.txt">N</a>',
            '<a href="https://external.example/x.diaml">B</a>',
            '<a href="/data/dialogue.diaml">Duplicate</a>',
            "No links",
        ]
    )
    monkeypatch.setattr(dialogbank_iso, "fetch_text", lambda url: next(pages))
    assert dialogbank_iso.discover_diaml_urls() == [
        "https://dialogbank.lsv.uni-saarland.de/data/dialogue.diaml",
        "https://external.example/x.diaml",
    ]


@pytest.mark.parametrize("reader", [coarse_discourse, dbdc3, dialogbank_iso])
def test_corpus_download_uses_declared_endpoint_and_request_headers(
    reader: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    response.read.return_value = b"fixture"
    opened = []

    def fake_open(request, timeout):
        opened.append((request.full_url, request.get_header("User-agent"), timeout))
        return response

    monkeypatch.setattr(reader.urllib.request, "urlopen", fake_open)
    if reader is dialogbank_iso:
        assert reader.fetch_text("https://example.test/p") == "fixture"
    else:
        assert reader.download_archive() == b"fixture"
    assert opened[0][1]
    assert opened[0][2] > 0
