from __future__ import annotations

import html.parser
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass

DIALOGBANK_BASE = "https://dialogbank.lsv.uni-saarland.de/"
DIALOGBANK_ENGLISH_PAGES = (
    "https://dialogbank.lsv.uni-saarland.de/?page_id=90",   # HCRC MapTask
    "https://dialogbank.lsv.uni-saarland.de/?page_id=92",   # Switchboard
    "https://dialogbank.lsv.uni-saarland.de/?page_id=312",  # DBOX
    "https://dialogbank.lsv.uni-saarland.de/?page_id=280",  # TRAINS
)

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


class _LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.links.append(value)


def fetch_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "eq-layer-dialogbank-audit/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def discover_diaml_urls() -> list[str]:
    urls: set[str] = set()
    for page in DIALOGBANK_ENGLISH_PAGES:
        parser = _LinkParser()
        parser.feed(fetch_text(page))
        for href in parser.links:
            absolute = urllib.parse.urljoin(DIALOGBANK_BASE, href)
            if absolute.lower().endswith(".diaml"):
                urls.add(absolute)
    return sorted(urls)


@dataclass(frozen=True)
class ISOExample:
    dialogue: str
    text: str
    function: str
    dimension: str


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _strip_ref(value: str | None) -> str:
    return (value or "").lstrip("#")


def parse_diaml(xml_text: str, source_url: str) -> list[ISOExample]:
    root = ET.fromstring(xml_text)

    words: dict[str, str] = {}
    for elem in root.iter():
        if _local(elem.tag) != "w":
            continue
        wid = elem.attrib.get(XML_ID)
        if wid:
            words[wid] = "".join(elem.itertext()).strip()

    verbal_segments: dict[str, list[str]] = {}
    for elem in root.iter():
        if _local(elem.tag) != "spanGrp" or elem.attrib.get("type") != "functionalVerbalSegment":
            continue
        sid = elem.attrib.get(XML_ID)
        if not sid:
            continue
        ids = []
        for span in elem:
            if _local(span.tag) != "span":
                continue
            start = _strip_ref(span.attrib.get("from"))
            end = _strip_ref(span.attrib.get("to"))
            if start:
                ids.append(start)
            if end and end != start:
                ids.append(end)
        verbal_segments[sid] = ids

    functional_to_verbal: dict[str, str] = {}
    for elem in root.iter():
        if _local(elem.tag) != "fs" or elem.attrib.get("type") != "functionalSegment":
            continue
        fsid = elem.attrib.get(XML_ID)
        if not fsid:
            continue
        for child in elem:
            if _local(child.tag) == "f" and child.attrib.get("name") == "verbalComponent":
                functional_to_verbal[fsid] = _strip_ref(child.attrib.get("fVal"))

    examples: list[ISOExample] = []
    for elem in root.iter():
        if _local(elem.tag) != "dialogueAct":
            continue
        function = elem.attrib.get("communicativeFunction")
        target = _strip_ref(elem.attrib.get("target"))
        if not function or not target:
            continue
        verbal = functional_to_verbal.get(target)
        token_ids = verbal_segments.get(verbal or "", [])
        text = " ".join(words.get(token, "") for token in token_ids).strip()
        if not text:
            continue
        examples.append(
            ISOExample(
                dialogue=source_url,
                text=text,
                function=function,
                dimension=elem.attrib.get("dimension", ""),
            )
        )
    return examples


def load_dialogbank_english() -> list[ISOExample]:
    examples: list[ISOExample] = []
    for url in discover_diaml_urls():
        examples.extend(parse_diaml(fetch_text(url), url))
    return examples


def inventory(examples: list[ISOExample]) -> dict:
    counts = Counter(example.function for example in examples)
    return {
        "dialogues": len({example.dialogue for example in examples}),
        "examples": len(examples),
        "functions": dict(sorted(counts.items())),
    }
