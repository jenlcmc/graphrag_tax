"""Parse IRS XML publications and form instructions.

All IRS XML files follow one of two schemas but share a common structural
pattern: sections are elements that have an `id` attribute and a `<hd>` child
element. Paragraph content lives in `<p>` and `<iconpara>` elements.

Supported schemas:
  tipx  (tips.xsd)   — IRS Publications (Pub 17, Pub 501, Pub 596, etc.)
  instrx (instrs.xsd) — Form instructions (i1040sc, i1040sb, i1040sd, etc.)
"""

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET


# Human-readable labels for each known IRS source key.
# Used to build the section_id string: "<Label>: <heading>"
SOURCE_LABELS: dict[str, str] = {
    # Publications
    "p17":   "IRS Pub. 17",
    "p501":  "IRS Pub. 501",
    "p502":  "IRS Pub. 502",
    "p503":  "IRS Pub. 503",
    "p504":  "IRS Pub. 504",
    "p505":  "IRS Pub. 505",
    "p523":  "IRS Pub. 523",
    "p525":  "IRS Pub. 525",
    "p526":  "IRS Pub. 526",
    "p529":  "IRS Pub. 529",
    "p530":  "IRS Pub. 530",
    "p544":  "IRS Pub. 544",
    "p559":  "IRS Pub. 559",
    "p560":  "IRS Pub. 560",
    "p570":  "IRS Pub. 570",
    "p596":  "IRS Pub. 596",
    "p936":  "IRS Pub. 936",
    "p970":  "IRS Pub. 970",
    "p1099": "IRS Pub. 1099",
    # Form instructions
    "i1040gi":  "Form 1040 Instructions",
    "i1040s8":  "Sch. 8812 Instructions",
    "i1040sb":  "Sch. B Instructions",
    "i1040sc":  "Sch. C Instructions",
    "i1040sca": "Sch. CA Instructions",
    "i1040sd":  "Sch. D Instructions",
    "i1040se":  "Sch. E Instructions",
    "i1040sse": "Sch. SE Instructions",
    "f1040es":  "Form 1040-ES Instructions",
    "i2441":    "Form 2441 Instructions",
    "i4562":    "Form 4562 Instructions",
    "i8606":    "Form 8606 Instructions",
    "i8829":    "Form 8829 Instructions",
    "i8863":    "Form 8863 Instructions",
    "i8949":    "Form 8949 Instructions",
    "i8995":    "Form 8995 Instructions",
    "i1099ac":  "Form 1099-AC Instructions",
    "i1099b":   "Form 1099-B Instructions",
    "i1099cap": "Form 1099-CAP Instructions",
    "i1099g":   "Form 1099-G Instructions",
    "i1099gi":  "Form 1099 General Instructions",
    "i1099h":   "Form 1099-H Instructions",
    "i1099int": "Form 1099-INT Instructions",
    "i1099k":   "Form 1099-K Instructions",
    "i1099ltc": "Form 1099-LTC Instructions",
    "i1099msc": "Form 1099-MISC Instructions",
    "i1099ptr": "Form 1099-PATR Instructions",
    "i1099q":   "Form 1099-Q Instructions",
    "i1099qa":  "Form 1099-QA Instructions",
    "i1099r":   "Form 1099-R Instructions",
    "i1099s":   "Form 1099-S Instructions",
    "i1099sa":  "Form 1099-SA Instructions",
}

# Minimum text length to keep a chunk (filters out empty or stub sections)
MIN_CHUNK_CHARS = 100


def parse(xml_path: Path, source: str) -> list[dict]:
    """Return a list of chunks from an IRS XML publication or form instruction file.

    ``source`` is the source key (e.g. "p17", "i1040sc"). It is used to look
    up the human-readable label in SOURCE_LABELS and stored on every chunk.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    ns          = _get_namespace(root)
    heading_tag = f"{{{ns}}}hd" if ns else "hd"
    para_tags   = {f"{{{ns}}}{t}" for t in ("p", "iconpara")} if ns else {"p", "iconpara"}

    chunks: list[dict] = []
    seen_section_ids: set[str] = set()
    _collect_sections(
        root,
        heading_tag,
        para_tags,
        source,
        chunks,
        parent_id=None,
        seen_section_ids=seen_section_ids,
    )
    return chunks


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------

def _get_namespace(root) -> str:
    if "}" in root.tag:
        return root.tag.split("}")[0].strip("{")
    return ""


def _collect_sections(
    element,
    heading_tag: str,
    para_tags: set,
    source: str,
    chunks: list[dict],
    parent_id: str | None,
    seen_section_ids: set[str],
) -> None:
    """Depth-first walk. Each element with an id + <hd> child + <p> content becomes a chunk."""
    element_id = element.get("id", "")
    heading_el = element.find(heading_tag)
    child_parent_id = parent_id

    if element_id and heading_el is not None:
        heading = "".join(heading_el.itertext()).strip()

        # Collect direct paragraph children only (not from nested sub-sections).
        # This keeps each chunk scoped to its own level in the document tree.
        paragraphs = [
            "".join(child.itertext()).strip()
            for child in element
            if child.tag in para_tags
        ]
        paragraphs = [p for p in paragraphs if p]

        if paragraphs:
            text = " ".join(paragraphs)
            if len(text) >= MIN_CHUNK_CHARS:
                chunk = _make_chunk(
                    element_id,
                    heading,
                    text,
                    source,
                    parent_id=parent_id,
                    seen_section_ids=seen_section_ids,
                )
                chunks.append(chunk)
                # Use the newly emitted section as parent for descendant chunks.
                child_parent_id = chunk["section_id"]

    for child in element:
        _collect_sections(
            child,
            heading_tag,
            para_tags,
            source,
            chunks,
            parent_id=child_parent_id,
            seen_section_ids=seen_section_ids,
        )


def _make_chunk(
    element_id: str,
    heading: str,
    text: str,
    source: str,
    parent_id: str | None,
    seen_section_ids: set[str],
) -> dict:
    label      = SOURCE_LABELS.get(source, source.upper())
    section_stub = heading[:80] if heading else element_id
    base_section_id = f"{label}: {section_stub}"
    section_id = _unique_section_id(base_section_id, element_id, seen_section_ids)
    chunk_id   = hashlib.md5(f"{source}:{element_id}".encode()).hexdigest()[:12]

    return {
        "id":         chunk_id,
        "section_id": section_id,
        "source":     source,
        "title":      heading,
        "text":       text,
        "hierarchy":  f"{source}/{element_id}",
        "parent_id":  parent_id,
        "cross_refs": [],  # populated later by normalizer
    }


def _unique_section_id(base: str, element_id: str, seen_section_ids: set[str]) -> str:
    """Return a deterministic unique section_id for graph/index node keys."""
    if base not in seen_section_ids:
        seen_section_ids.add(base)
        return base

    with_element = f"{base} [{element_id}]"
    if with_element not in seen_section_ids:
        seen_section_ids.add(with_element)
        return with_element

    suffix = 2
    while True:
        candidate = f"{with_element}#{suffix}"
        if candidate not in seen_section_ids:
            seen_section_ids.add(candidate)
            return candidate
        suffix += 1
