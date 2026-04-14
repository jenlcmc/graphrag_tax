"""Parse plain-text SARA statute source files into retrieval chunks.

Expected input layout:
  knowledge/<profile>/source/section151
  knowledge/<profile>/source/section63
  ...
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


_SECTION_FROM_NAME_RE = re.compile(r"section\s*([0-9]{1,4}[A-Z]?)", re.IGNORECASE)
_SECTION_FROM_TITLE_RE = re.compile(r"§\s*([0-9]{1,4}[A-Z]?)")


def _canonical_section_ref(file_name: str, text: str) -> str | None:
    match = _SECTION_FROM_NAME_RE.search(file_name)
    if not match:
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        match = _SECTION_FROM_TITLE_RE.search(first_line)
    if not match:
        return None
    return f"26 USC §{match.group(1)}"


def parse(source_path: Path, source: str = "sara_source") -> list[dict]:
    """Return chunks from a SARA source directory or a single source file."""
    if source_path.is_file():
        files = [source_path]
    elif source_path.is_dir():
        files = [path for path in sorted(source_path.iterdir()) if path.is_file()]
    else:
        return []

    chunks: list[dict] = []
    for path in files:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        text = raw_text.strip()
        if not text:
            continue

        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        section_ref = _canonical_section_ref(path.name, text)

        if section_ref:
            section_id = f"SARA Source: {section_ref}"
            title = first_line or section_ref
            cross_refs = [section_ref]
        else:
            section_id = f"SARA Source: {path.name}"
            title = first_line or path.name
            cross_refs = []

        chunk_id = hashlib.md5(f"{source}:{path.name}".encode("utf-8")).hexdigest()[:12]
        chunks.append(
            {
                "id": chunk_id,
                "section_id": section_id,
                "source": source,
                "title": title,
                "text": text,
                "hierarchy": f"{source}/{path.name}",
                "parent_id": None,
                "cross_refs": cross_refs,
            }
        )

    return chunks
