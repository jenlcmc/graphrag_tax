"""Utilities for matching citation/retrieval references with hierarchical USC awareness."""

from __future__ import annotations

import re


_USC_REF_RE = re.compile(r"^26\s+usc\s*§\s*([0-9]{1,4}[a-z]?)(.*)$", re.IGNORECASE)
_SUBSECTION_RE = re.compile(r"\(([a-z0-9]+)\)", re.IGNORECASE)


def _normalize_ref(ref: str) -> str:
    return re.sub(r"\s+", " ", (ref or "").strip().lower())


def parse_usc_reference(ref: str) -> tuple[str, tuple[str, ...]] | None:
    """Parse canonical USC ref into (section, subsections).

    Example: "26 USC §151(d)(1)" -> ("151", ("d", "1"))
    """
    normalized = _normalize_ref(ref)
    match = _USC_REF_RE.match(normalized)
    if not match:
        return None

    section = match.group(1)
    suffix = match.group(2) or ""
    subsections = tuple(_SUBSECTION_RE.findall(suffix))
    return (section, subsections)


def _is_prefix(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return len(left) <= len(right) and left == right[: len(left)]


def _common_prefix_len(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    n = 0
    for l_val, r_val in zip(left, right):
        if l_val != r_val:
            break
        n += 1
    return n


def reference_match_score(reference: str, target: str) -> float:
    """Return match score in [0,1] between two references.

    - 1.0  : exact match
    - 0.75 : direct parent/child subsection relation
    - 0.5  : ancestor/descendant with depth gap >= 2
    - 0.25 : same section with shared subsection stem (sibling-ish)
    - 0.0  : no relation
    """
    ref_norm = _normalize_ref(reference)
    tgt_norm = _normalize_ref(target)
    if not ref_norm or not tgt_norm:
        return 0.0

    if ref_norm == tgt_norm:
        return 1.0

    ref_parsed = parse_usc_reference(ref_norm)
    tgt_parsed = parse_usc_reference(tgt_norm)
    if not ref_parsed or not tgt_parsed:
        return 0.0

    ref_section, ref_sub = ref_parsed
    tgt_section, tgt_sub = tgt_parsed
    if ref_section != tgt_section:
        return 0.0

    if _is_prefix(ref_sub, tgt_sub) or _is_prefix(tgt_sub, ref_sub):
        depth_delta = abs(len(ref_sub) - len(tgt_sub))
        return 0.75 if depth_delta == 1 else 0.5

    if _common_prefix_len(ref_sub, tgt_sub) > 0:
        return 0.25

    return 0.0


def best_match_score(reference: str, targets: list[str]) -> tuple[float, str | None]:
    """Return (best_score, best_target) over candidate targets."""
    best_score = 0.0
    best_target = None
    for target in targets:
        score = reference_match_score(reference, target)
        if score > best_score:
            best_score = score
            best_target = target
    return (best_score, best_target)
