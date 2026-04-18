"""Parse plain-text SARA statute source files into hierarchical retrieval chunks.

Expected input layout:
  knowledge/<profile>/source/section151
  knowledge/<profile>/source/section63
  ...

Each file is split into a section-level chunk (full text, parent_id=None) plus
one chunk per top-level subsection (a), (b), … (parent_id = section chunk id).
Subsection chunks retain their paragraphs / sub-paragraphs inline so retrievers
get self-contained text, but the section chunk is also stored for graph
community detection and broad queries.

Hierarchy depth recognised by indentation (spaces):
  0  → subsection   (a), (b), …
  4  → paragraph    (1), (2), …
  8  → subparagraph (A), (B), …
  12 → clause       (i), (ii), …
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


_SECTION_FROM_NAME_RE = re.compile(r"section\s*([0-9]{1,4}[A-Z]?)", re.IGNORECASE)
_SECTION_FROM_TITLE_RE = re.compile(r"§\s*([0-9]{1,4}[A-Z]?)")
_SUBSECTION_RE = re.compile(r"^\(([a-zA-Z])\)\s")


def _extract_section_num(file_name: str, text: str) -> str | None:
    m = _SECTION_FROM_NAME_RE.search(file_name)
    if not m:
        first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
        m = _SECTION_FROM_TITLE_RE.search(first_line)
    return m.group(1) if m else None


def _chunk_id(source: str, file_name: str, suffix: str) -> str:
    return hashlib.md5(f"{source}:{file_name}:{suffix}".encode()).hexdigest()[:12]


def _subsection_title(subsec_text: str) -> str:
    """Return the heading text of a subsection block (first non-empty line)."""
    for line in subsec_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _split_subsections(lines: list[str]) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Split file lines into (intro_lines, [(letter, subsec_lines), ...])."""
    intro: list[str] = []
    subsections: list[tuple[str, list[str]]] = []
    current_letter: str | None = None
    current_lines: list[str] = []

    for line in lines:
        m = _SUBSECTION_RE.match(line)
        if m and not line.startswith(" "):
            if current_letter is not None:
                subsections.append((current_letter, current_lines))
            else:
                intro = current_lines
            current_letter = m.group(1).lower()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_letter is not None:
        subsections.append((current_letter, current_lines))
    elif not subsections:
        intro = current_lines

    return intro, subsections


def parse(source_path: Path, source: str = "sara_source") -> list[dict]:
    """Return hierarchical chunks from a SARA source directory or single file."""
    if source_path.is_file():
        files = [source_path]
    elif source_path.is_dir():
        files = [p for p in sorted(source_path.iterdir()) if p.is_file()]
    else:
        return []

    chunks: list[dict] = []
    for path in files:
        raw_text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw_text:
            continue

        section_num = _extract_section_num(path.name, raw_text)
        section_ref = f"26 USC §{section_num}" if section_num else None
        section_id = f"SARA Source: {section_ref}" if section_ref else f"SARA Source: {path.name}"
        title_line = next((l.strip() for l in raw_text.splitlines() if l.strip()), path.name)

        # Section-level chunk: full text, no parent.
        chunks.append(
            {
                "id": _chunk_id(source, path.name, "section"),
                "section_id": section_id,
                "source": source,
                "title": title_line,
                "text": raw_text,
                "hierarchy": f"{source}/{path.name}",
                "parent_id": None,
                "cross_refs": [section_ref] if section_ref else [],
            }
        )

        lines = raw_text.splitlines()
        intro_lines, subsections = _split_subsections(lines)
        intro_text = "\n".join(intro_lines).strip()

        for letter, sub_lines in subsections:
            subsec_text = "\n".join(sub_lines).strip()
            subsec_ref = f"26 USC §{section_num}({letter})" if section_num else None
            subsec_section_id = (
                f"SARA Source: {subsec_ref}" if subsec_ref else f"{section_id}/({letter})"
            )

            # Include section header + intro for self-contained context.
            context_parts = [title_line]
            if intro_text:
                context_parts.append(intro_text)
            context_parts.append(subsec_text)
            full_text = "\n\n".join(context_parts)

            sub_title = _subsection_title(subsec_text)
            cross_refs = []
            if section_ref:
                cross_refs.append(section_ref)
            if subsec_ref:
                cross_refs.append(subsec_ref)

            chunks.append(
                {
                    "id": _chunk_id(source, path.name, letter),
                    "section_id": subsec_section_id,
                    "source": source,
                    "title": sub_title,
                    "text": full_text,
                    "hierarchy": f"{source}/{path.name}/({letter})",
                    "parent_id": section_id,  # must be section_id string, not chunk MD5
                    "cross_refs": cross_refs,
                }
            )

    return chunks
