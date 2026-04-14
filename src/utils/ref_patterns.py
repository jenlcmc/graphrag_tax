"""Shared regex patterns for extracting USC and IRS references from text.

Imported by both src/preprocessing/normalizer.py (build time) and
src/retrieval/graph_retriever.py (query time) so that the two modules
stay in sync without duplicating pattern definitions.
"""

import re

# ---------------------------------------------------------------------------
# USC section reference patterns
# ---------------------------------------------------------------------------

# Matches "section 401(k)", "§ 61(b)(2)", "§199A".
# Captures the section number and up to four nested subsection levels.
USC_SECTION_RE = re.compile(
    r"(?:section\s+|§\s*)(\d{1,4}[A-Z]?)(?:((?:\([a-z0-9]+\)){1,4}))?",
    re.IGNORECASE,
)

# Matches section ranges like "§101-108" or "section 1211-1212".
USC_SECTION_RANGE_RE = re.compile(
    r"(?:section\s+|§\s*)(\d{1,4})\s*[-–]\s*(\d{1,4})",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# IRS publication / form / schedule reference patterns
# ---------------------------------------------------------------------------

# Matches "Publication 596", "Pub. 17", "pub 501".
PUB_RE = re.compile(r"(?:publication|pub\.?)\s+([0-9]{1,4}[a-z]?)", re.IGNORECASE)

# Matches "Form 2441", "Form 1040-ES", "Form 1099-Q".
FORM_RE = re.compile(r"form\s+([0-9]{3,5}(?:-[a-z0-9]{1,4})?)", re.IGNORECASE)

# Matches "Schedule C", "Sch. SE", "Schedule 8812", "Schedule EIC".
SCHEDULE_RE = re.compile(
    r"(?:schedule|sch\.?)\s+([a-z]{1,4}|[0-9]{1,4})",
    re.IGNORECASE,
)

# Combined form/schedule detector used for quick boolean checks.
FORM_SCHEDULE_RE = re.compile(
    r"schedule\s+(?:se|eic|8812|[a-e])|form\s+(?:1040(?:-es)?|1099-q|\d{4})",
    re.IGNORECASE,
)
