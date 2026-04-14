"""NLP normalization: extract cross-references from IRS explanation chunks.

This module now derives IRS reference mappings from SOURCE_LABELS so reference
coverage scales automatically to all known publications/forms/schedules in
knowledge/ without hand-maintaining per-source dictionaries.
"""

import re

from src.ingestion.irs_xml_parser import SOURCE_LABELS
from src.utils.ref_patterns import (
    USC_SECTION_RE as _USC_SECTION_RE,
    USC_SECTION_RANGE_RE as _USC_SECTION_RANGE_RE,
    PUB_RE as _PUB_RE_BASE,
    FORM_RE as _FORM_RE_BASE,
    SCHEDULE_RE as _SCHEDULE_RE_BASE,
)


def extract_usc_refs(text: str) -> list[str]:
    """Return canonical 26 USC citations found in text."""
    refs: set[str] = set()

    # Expand compact ranges so graph linking can follow every cited section.
    for start_raw, end_raw in _USC_SECTION_RANGE_RE.findall(text):
        start = int(start_raw)
        end = int(end_raw)
        if start <= end and (end - start) <= 60:
            for value in range(start, end + 1):
                refs.add(f"26 USC §{value}")

    for match in _USC_SECTION_RE.finditer(text):
        section_num = match.group(1)
        suffix_raw = match.group(2) or ""
        parts = re.findall(r"\(([a-z0-9]+)\)", suffix_raw, flags=re.IGNORECASE)
        suffix = "".join(f"({part})" for part in parts)
        refs.add(f"26 USC §{section_num}{suffix}")

    return sorted(refs)


# ---------------------------------------------------------------------------
# IRS publication/form/schedule reference extraction
# ---------------------------------------------------------------------------

_PUB_RE      = _PUB_RE_BASE
_FORM_RE     = _FORM_RE_BASE
_SCHEDULE_RE = _SCHEDULE_RE_BASE


def _normalize_key(text: str) -> str:
    text = text.lower().strip()
    text = text.replace(".", "")
    text = re.sub(r"\s+", " ", text)
    return text


def _build_publication_label_map() -> dict[str, str]:
    """Map publication tokens (e.g. "596") to section-id label prefixes."""
    mapping: dict[str, str] = {}
    for label in SOURCE_LABELS.values():
        match = re.fullmatch(r"IRS Pub\.\s+([0-9]{1,4}[A-Z]?)", label)
        if match:
            mapping[match.group(1).lower()] = label
    return mapping


def _build_form_schedule_label_map() -> dict[str, str]:
    """Build alias map for all known form/schedule labels in SOURCE_LABELS."""
    mapping: dict[str, str] = {}

    for label in SOURCE_LABELS.values():
        if label.startswith("Form ") and label.endswith(" Instructions"):
            code = label[len("Form ") : -len(" Instructions")].strip()
            key = _normalize_key(f"form {code}")
            mapping[key] = label
            mapping[_normalize_key(key.replace("-", ""))] = label

        if label.startswith("Sch. ") and label.endswith(" Instructions"):
            code = label[len("Sch. ") : -len(" Instructions")].strip()
            key = _normalize_key(f"schedule {code}")
            mapping[key] = label
            mapping[_normalize_key(f"sch {code}")] = label

    # Schedule EIC is part of Form 1040 instructions in this corpus.
    if "i1040gi" in SOURCE_LABELS:
        form_1040_label = SOURCE_LABELS["i1040gi"]
        mapping[_normalize_key("schedule eic")] = form_1040_label
        mapping[_normalize_key("sch eic")] = form_1040_label

    return mapping


_PUB_LABEL = _build_publication_label_map()
_FORM_SCHEDULE_LABEL = _build_form_schedule_label_map()


def extract_irs_refs(text: str) -> list[str]:
    """Return section-id label prefixes for IRS refs found in text."""
    refs: set[str] = set()

    for match in _PUB_RE.finditer(text):
        pub_token = match.group(1).lower()
        label = _PUB_LABEL.get(pub_token)
        if label:
            refs.add(label)

    for match in _FORM_RE.finditer(text):
        form_code = match.group(1)
        key = _normalize_key(f"form {form_code}")
        label = _FORM_SCHEDULE_LABEL.get(key)
        if label:
            refs.add(label)
            continue

        # Fallback for mentions that omit the hyphen (e.g. "Form 1099Q").
        key_no_hyphen = _normalize_key(f"form {form_code.replace('-', '')}")
        label = _FORM_SCHEDULE_LABEL.get(key_no_hyphen)
        if label:
            refs.add(label)

    for match in _SCHEDULE_RE.finditer(text):
        schedule_code = match.group(1)
        key = _normalize_key(f"schedule {schedule_code}")
        label = _FORM_SCHEDULE_LABEL.get(key)
        if label:
            refs.add(label)

    return sorted(refs)


# ---------------------------------------------------------------------------
# Main annotation function
# ---------------------------------------------------------------------------


def load_nlp(model: str = "en_core_web_sm"):
    """Load and return a spaCy language model when available.

    The normalizer currently uses regex extraction only, so spaCy is optional.
    If spaCy is unavailable or misconfigured in the active environment, return
    ``None`` and let the pipeline continue.
    """
    try:
        import spacy

        return spacy.load(model)
    except Exception as exc:
        print(f"Warning: spaCy model load skipped ({exc})")
        return None


def annotate_chunks(chunks: list[dict], nlp) -> list[dict]:
    """Add cross_refs to IRS chunks by extracting USC and IRS references.

    The ``nlp`` argument is kept for pipeline API compatibility.
    """
    for chunk in chunks:
        if chunk["source"] == "usc26":
            continue

        text = chunk["text"]
        usc_refs = extract_usc_refs(text)
        irs_refs = extract_irs_refs(text)

        existing = set(chunk.get("cross_refs", []))
        chunk["cross_refs"] = sorted(existing | set(usc_refs) | set(irs_refs))

    return chunks
