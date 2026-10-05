from __future__ import annotations

import html.parser
import re
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


def _repair_xml_text(xml_text: str) -> str:
    # Some legacy DialogBank files contain bare ampersands or control
    # characters in transcribed speech. Repair only characters that XML 1.0
    # cannot represent; do not alter annotation structure or labels.
    xml_text = re.sub(
        r"&(?!#\d+;|#x[0-9A-Fa-f]+;|[A-Za-z_][A-Za-z0-9_.:-]*;)",
        "&amp;",
        xml_text,
    )
    return "".join(
        ch
        for ch in xml_text
        if ch in "\t\n\r" or ord(ch) >= 0x20
    )



def _parse_attrs(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key, _quote, value in re.findall(
        r"([A-Za-z_][\\w:.-]*)\\s*=\\s*(['\"])(.*?)\\2",
        raw,
        flags=re.DOTALL,
    ):
        attrs[key] = html.unescape(value)
    return attrs


def _plain_xml_text(raw: str) -> str:
    without_tags = re.sub(r"<[^>]+>", "", raw)
    return " ".join(html.unescape(without_tags).split())


def _parse_diaml_legacy(xml_text: str, source_url: str) -> list[ISOExample]:
    """Extract the annotation graph without requiring globally valid XML.

    A few legacy DialogBank files contain malformed transcript characters.
    The annotation tags themselves are regular enough to recover the fields
    EQ-Layer needs. This fallback never rewrites or fabricates labels.
    """
    words: dict[str, str] = {}
    for match in re.finditer(r"<(?:[A-Za-z_][\\w.-]*:)?w\\b([^>]*)>(.*?)</(?:[A-Za-z_][\\w.-]*:)?w>", xml_text, flags=re.DOTALL):
        attrs = _parse_attrs(match.group(1))
        wid = attrs.get("xml:id")
        if wid:
            words[wid] = _plain_xml_text(match.group(2))

    verbal_segments: dict[str, list[str]] = {}
    for match in re.finditer(
        r"<(?:[A-Za-z_][\\w.-]*:)?spanGrp\\b([^>]*)>(.*?)</(?:[A-Za-z_][\\w.-]*:)?spanGrp>",
        xml_text,
        flags=re.DOTALL,
    ):
        attrs = _parse_attrs(match.group(1))
        if attrs.get("type") != "functionalVerbalSegment":
            continue
        sid = attrs.get("xml:id")
        if not sid:
            continue
        ids: list[str] = []
        for span in re.finditer(r"<(?:[A-Za-z_][\\w.-]*:)?span\\b([^>]*)/?>", match.group(2)):
            span_attrs = _parse_attrs(span.group(1))
            start = _strip_ref(span_attrs.get("from"))
            end = _strip_ref(span_attrs.get("to"))
            if start:
                ids.append(start)
            if end and end != start:
                ids.append(end)
        verbal_segments[sid] = ids

    functional_to_verbal: dict[str, str] = {}
    for match in re.finditer(r"<(?:[A-Za-z_][\\w.-]*:)?fs\\b([^>]*)>(.*?)</(?:[A-Za-z_][\\w.-]*:)?fs>", xml_text, flags=re.DOTALL):
        attrs = _parse_attrs(match.group(1))
        if attrs.get("type") != "functionalSegment":
            continue
        fsid = attrs.get("xml:id")
        if not fsid:
            continue
        for fmatch in re.finditer(r"<(?:[A-Za-z_][\\w.-]*:)?f\\b([^>]*)/?>", match.group(2)):
            fattrs = _parse_attrs(fmatch.group(1))
            if fattrs.get("name") == "verbalComponent":
                functional_to_verbal[fsid] = _strip_ref(fattrs.get("fVal"))
                break

    examples: list[ISOExample] = []
    for match in re.finditer(r"<(?:[A-Za-z_][\\w.-]*:)?dialogueAct\\b([^>]*)/?>", xml_text):
        attrs = _parse_attrs(match.group(1))
        function = attrs.get("communicativeFunction")
        target = _strip_ref(attrs.get("target"))
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
                dimension=attrs.get("dimension", ""),
            )
        )

    if not examples:
        raise ValueError(f"Legacy DialogBank parser recovered no dialogue acts: {source_url}")
    return examples


def parse_diaml(xml_text: str, source_url: str) -> list[ISOExample]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        try:
            root = ET.fromstring(_repair_xml_text(xml_text))
        except ET.ParseError:
            return _parse_diaml_legacy(xml_text, source_url)

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
