"""Parse IRS publication and instruction PDFs into chunk records.

This parser is used when XML sources are unavailable (for example, year-profile
folders that only contain IRS PDFs). Output schema matches the XML parsers so
the downstream normalizer, chunker, vector index, and graph index can run
without changes.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from src import config as cfg
from src.ingestion.irs_xml_parser import SOURCE_LABELS


_HEADING_RE = re.compile(
    r"^(part\s+[ivxlcdm0-9]+|chapter\s+\d+|section\s+\d+|line\s+\d+|"
    r"general instructions|specific instructions|worksheet|purpose of|who must|"
    r"what'?s new|definitions|exceptions?)\b",
    re.IGNORECASE,
)


def parse(pdf_path: Path, source: str) -> list[dict]:
    """Return chunk list from an IRS PDF using section-style extraction."""
    reader = PdfReader(str(pdf_path))
    chunks: list[dict] = []
    seen_section_ids: set[str] = set()
    parser_version = "pdf_section_v2"

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        lines = _clean_lines(raw_text)
        if not lines:
            continue

        sections = _extract_sections(lines, page_num)
        for section_idx, (heading, section_text) in enumerate(sections, start=1):
            windows = _split_long_text(section_text, cfg.PDF_SECTION_MAX_CHARS, cfg.PDF_SECTION_OVERLAP_CHARS)

            for window_idx, text in enumerate(windows, start=1):
                if len(text) < cfg.PDF_SECTION_MIN_CHARS and len(windows) > 1:
                    continue

                title = _guess_title(heading, text, source, page_num)
                section_base = (
                    f"{_source_label(source)}: p{page_num:03d} "
                    f"s{section_idx:02d} {title}"
                )
                section_id = _unique_section_id(section_base, page_num, seen_section_ids)
                chunk_id = hashlib.md5(
                    f"{source}:{pdf_path.name}:p{page_num}:s{section_idx}:w{window_idx}".encode()
                ).hexdigest()[:12]

                chunks.append(
                    {
                        "id": chunk_id,
                        "section_id": section_id,
                        "source": source,
                        "title": title,
                        "text": text,
                        "hierarchy": f"{source}/page/{page_num}/section/{section_idx}",
                        "parent_id": None,
                        "cross_refs": [],  # populated later by normalizer
                        "source_file": pdf_path.name,
                        "page_start": page_num,
                        "page_end": page_num,
                        "chunk_method": "pdf_section",
                        "metadata": {
                            "parser": parser_version,
                            "source_file": pdf_path.name,
                            "page": page_num,
                            "section_index": section_idx,
                            "window_index": window_idx,
                            "section_heading": heading,
                        },
                    }
                )

    return chunks


def _source_label(source: str) -> str:
    return SOURCE_LABELS.get(source, source.upper())


def _normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return " ".join(lines)


def _clean_lines(text: str) -> list[str]:
    if not text.strip():
        return []
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return [line for line in lines if line]


def _is_heading(line: str) -> bool:
    if not line:
        return False
    if len(line) > 110:
        return False
    if line.endswith(":"):
        return True
    if _HEADING_RE.match(line):
        return True

    letters = [ch for ch in line if ch.isalpha()]
    if len(letters) >= 6:
        uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        if uppercase_ratio >= 0.85:
            return True

    return False


def _extract_sections(lines: list[str], page_num: int) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = f"Page {page_num}"
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        text = _normalize_text("\n".join(current))
        if len(text) >= cfg.PDF_SECTION_MIN_CHARS:
            sections.append((heading, text))

    for line in lines:
        if _is_heading(line) and current:
            flush()
            current = []
            heading = line[:120]
            if len(sections) >= cfg.PDF_MAX_SECTIONS_PER_PAGE:
                break
            continue
        current.append(line)

    if len(sections) < cfg.PDF_MAX_SECTIONS_PER_PAGE:
        flush()

    if not sections:
        text = _normalize_text("\n".join(lines))
        if len(text) >= cfg.PDF_SECTION_MIN_CHARS:
            sections.append((heading, text))

    return sections


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return _slice_with_overlap(text, max_chars, overlap_chars)

    windows: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        extra = len(sentence) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            window_text = " ".join(current)
            windows.append(window_text)

            overlap: list[str] = []
            overlap_len = 0
            for part in reversed(current):
                part_len = len(part) + (1 if overlap else 0)
                if overlap and overlap_len + part_len > overlap_chars:
                    break
                overlap.insert(0, part)
                overlap_len += part_len

            current = overlap[:]
            current_len = len(" ".join(current)) if current else 0

        if current:
            current.append(sentence)
            current_len += len(sentence) + 1
        else:
            current = [sentence]
            current_len = len(sentence)

    if current:
        windows.append(" ".join(current))

    return windows if windows else [text]


def _slice_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    windows: list[str] = []
    start = 0
    step = max(1, max_chars - max(0, overlap_chars))
    while start < len(text):
        window = text[start : start + max_chars].strip()
        if window:
            windows.append(window)
        start += step
    return windows or [text]


def _guess_title(heading: str, text: str, source: str, page_num: int) -> str:
    if heading and heading.strip():
        heading_clean = heading.strip()
        if len(heading_clean) <= 96:
            return heading_clean
        return heading_clean[:96]

    first_sentence = text.split(".", 1)[0].strip()
    if not first_sentence:
        return f"{_source_label(source)} Page {page_num}"

    words = first_sentence.split()
    if len(words) > 14:
        first_sentence = " ".join(words[:14])

    return first_sentence[:96]


def _unique_section_id(base: str, page_num: int, seen_section_ids: set[str]) -> str:
    if base not in seen_section_ids:
        seen_section_ids.add(base)
        return base

    with_page = f"{base} [p{page_num}]"
    if with_page not in seen_section_ids:
        seen_section_ids.add(with_page)
        return with_page

    suffix = 2
    while True:
        candidate = f"{with_page}#{suffix}"
        if candidate not in seen_section_ids:
            seen_section_ids.add(candidate)
            return candidate
        suffix += 1