"""Split oversized chunks at sentence boundaries.

Uses spaCy sentence boundaries when available, with a deterministic regex
fallback for environments where spaCy is unavailable.
"""

import re

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy


def split_chunk(chunk: dict, nlp, max_chars: int) -> list[dict]:
    """Return one or more chunks by splitting text at sentence boundaries.

    If the text is already within max_chars, returns the chunk unchanged.
    Splitting preserves all metadata fields; large sub-chunks that still
    exceed max_chars are kept as-is rather than split arbitrarily mid-sentence.
    """
    text = chunk["text"]
    if len(text) <= max_chars:
        return [chunk]

    sentences = _split_sentences(text, nlp)

    parts   = []
    current = []
    current_len = 0

    for sentence in sentences:
        if current and current_len + len(sentence) > max_chars:
            parts.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence)

    if current:
        parts.append(" ".join(current))

    result = []
    for i, part in enumerate(parts):
        sub = dict(chunk)
        sub["text"] = part
        sub["id"]   = f"{chunk['id']}_{i}"
        result.append(sub)

    return result


def _split_sentences(text: str, nlp) -> list[str]:
    """Split text into sentences using spaCy when possible.

    Falls back to punctuation-based splitting if ``nlp`` is ``None`` or if the
    spaCy pipeline raises an error.
    """
    if nlp is not None:
        try:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if sentences:
                return sentences
        except Exception:
            pass

    # Deterministic fallback: split on end punctuation followed by whitespace.
    fallback = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if fallback:
        return fallback

    return [text.strip()] if text.strip() else []


def apply_to_all(chunks: list[dict], nlp, max_chars: int) -> list[dict]:
    """Apply split_chunk to every chunk in the list."""
    result = []
    for chunk in chunks:
        result.extend(split_chunk(chunk, nlp, max_chars))
    return result
