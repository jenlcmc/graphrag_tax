"""Parse Title 26 (IRC) from USLM XML. Focuses on Subtitle A (individual income tax)."""

import re
import hashlib
from pathlib import Path
from lxml import etree

NS = "http://xml.house.gov/schemas/uslm/1.0"


def _tag(name: str) -> str:
    return f"{{{NS}}}{name}"


SKIP_TAGS = {
    _tag("sourceCredit"),
    _tag("notes"),
    _tag("note"),
    _tag("table"),
    _tag("toc"),
}


def parse(xml_path: Path, max_chunk_chars: int = 2000) -> list[dict]:
    """Return a list of chunks from Subtitle A of Title 26."""
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    subtitle_a = root.find(f".//{_tag('subtitle')}[@identifier='/us/usc/t26/stA']")
    if subtitle_a is None:
        raise ValueError("Subtitle A not found in usc26.xml")

    chunks = []
    for section in subtitle_a.iter(_tag("section")):
        identifier = section.get("identifier", "")
        if not _is_top_level_section(identifier):
            continue
        chunks.extend(_section_to_chunks(section, max_chunk_chars))

    return chunks


def to_canonical(identifier: str) -> str:
    """Convert a USLM identifier to a canonical citation string.

    Example: '/us/usc/t26/stA/ch1/s401/k/1' -> '26 USC §401(k)(1)'
    """
    parts = [p for p in identifier.split("/") if p]

    for i, part in enumerate(parts):
        if re.match(r"^s\d", part):
            section_num = part[1:]
            sub_parts = parts[i + 1:]
            suffix = "".join(f"({p})" for p in sub_parts)
            return f"26 USC §{section_num}{suffix}"

    return identifier


# --- private helpers ---


def _is_top_level_section(identifier: str) -> bool:
    # Match paths ending in /sNUM or /sNUMletter (e.g. /s1, /s401, /s23A)
    # Excludes subsections like /s1/a and structural paths like /stA
    return bool(re.search(r"/s\d+[A-Z]?$", identifier))


def _section_to_chunks(section, max_chars: int) -> list[dict]:
    identifier = section.get("identifier", "")
    section_id = to_canonical(identifier)
    heading    = _get_heading(section)
    full_text  = _extract_text(section)
    cross_refs = _get_cross_refs(section)

    if len(full_text) <= max_chars:
        return [_make_chunk(identifier, section_id, heading, full_text, None, cross_refs)]

    # Split at subsection boundaries when the section is too long
    chunks = []
    for subsection in section.findall(_tag("subsection")):
        sub_id       = subsection.get("identifier", "")
        sub_section_id = to_canonical(sub_id)
        sub_heading  = _get_heading(subsection)
        sub_text     = _extract_text(subsection)
        sub_refs     = _get_cross_refs(subsection)
        title        = f"{heading} - {sub_heading}" if sub_heading else heading
        chunks.append(_make_chunk(sub_id, sub_section_id, title, sub_text, section_id, sub_refs))

    # Fall back to the full section if no subsections found
    if not chunks:
        chunks = [_make_chunk(identifier, section_id, heading, full_text, None, cross_refs)]

    return chunks


def _get_heading(element) -> str:
    heading = element.find(_tag("heading"))
    if heading is not None and heading.text:
        return heading.text.strip()
    return ""


def _extract_text(element) -> str:
    parts = []
    _collect_text(element, parts)
    return " ".join(p for p in parts if p)


def _collect_text(element, parts: list[str]) -> None:
    if element.tag in SKIP_TAGS:
        return
    if element.text and element.text.strip():
        parts.append(element.text.strip())
    for child in element:
        _collect_text(child, parts)
        if child.tail and child.tail.strip():
            parts.append(child.tail.strip())


def _get_cross_refs(element) -> list[str]:
    refs = set()
    for ref in element.iter(_tag("ref")):
        href = ref.get("href", "")
        if href.startswith("/us/usc/t26/s"):
            refs.add(to_canonical(href))
    return sorted(refs)


def _make_chunk(
    identifier: str,
    section_id: str,
    title: str,
    text: str,
    parent_id: str | None,
    cross_refs: list[str],
) -> dict:
    chunk_id = hashlib.md5(identifier.encode()).hexdigest()[:12]
    return {
        "id":         chunk_id,
        "section_id": section_id,
        "source":     "usc26",
        "title":      title,
        "text":       text,
        "hierarchy":  identifier,
        "parent_id":  parent_id,
        "cross_refs": cross_refs,
    }
