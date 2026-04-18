"""BM25 sparse retriever.

Implements Okapi BM25 (Robertson & Zaragoza, 2009) with no external
dependencies.  BM25 complements dense vector search in two ways:
  1. Exact keyword recall  — section numbers like "§63", form names like
     "Schedule C", and domain jargon score reliably via term frequency.
  2. Orthogonality to dense  — BM25 errors and vector errors are largely
     uncorrelated, so RRF fusion of both consistently outperforms either
     alone (Rackauckas, "RAG-Fusion", arXiv:2402.03367).

The index is built in RAM from chunk metadata (no disk persistence needed)
since build time is O(total_tokens) and is fast enough to redo on each load.

BM25 parameters:
  k1 = 1.5   (term-frequency saturation; 1.2–2.0 typical)
  b  = 0.75  (document-length normalisation; 0.75 standard)
"""

from __future__ import annotations

import math
import re
from pathlib import Path


_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9§]{2,}\b")

_STOPWORDS = frozenset({
    "about", "after", "also", "and", "any", "are", "can", "for", "from",
    "how", "into", "its", "like", "line", "lines", "more", "that", "the",
    "their", "them", "then", "there", "these", "this", "those", "what",
    "when", "where", "which", "with", "would", "your", "per", "under",
    "shall", "such", "each", "its", "not", "may", "has", "have", "been",
})

_K1: float = 1.5
_B:  float = 0.75


def _tokenize(text: str) -> list[str]:
    return [
        t.lower() for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOPWORDS and len(t) >= 2
    ]


class SparseRetriever:
    """Okapi BM25 over chunk text + section_id + title for sparse keyword matching."""

    def __init__(self) -> None:
        self._metadata: list[dict] = []
        self._tf: list[dict[str, int]] = []
        self._dl: list[int] = []
        self._df: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avg_dl: float = 0.0
        self._n: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: list[dict]) -> None:
        """Index a list of chunk dicts (must have section_id, title, text)."""
        self._metadata = []
        self._tf = []
        self._dl = []
        self._df = {}

        for chunk in chunks:
            # Index section_id + title + text so that section references in
            # queries ("§63", "section 151") match even when not in body text.
            doc_text = " ".join([
                chunk.get("section_id", ""),
                chunk.get("title", ""),
                chunk.get("text", ""),
            ])
            tokens = _tokenize(doc_text)
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1

            self._metadata.append({
                "section_id": chunk["section_id"],
                "title":      chunk.get("title", ""),
                "source":     chunk.get("source", ""),
                "text":       chunk.get("text", ""),
                "hierarchy":  chunk.get("hierarchy"),
                "parent_id":  chunk.get("parent_id"),
                "cross_refs": chunk.get("cross_refs", []),
            })
            self._tf.append(tf)
            self._dl.append(len(tokens))
            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1

        self._n = len(chunks)
        self._avg_dl = sum(self._dl) / max(1, self._n)
        self._idf = {
            term: math.log((self._n - df + 0.5) / (df + 0.5) + 1)
            for term, df in self._df.items()
        }

    @classmethod
    def build_from_metadata(cls, metadata: list[dict]) -> "SparseRetriever":
        """Convenience constructor from VectorIndex.metadata (same schema)."""
        inst = cls()
        inst.build(metadata)
        return inst

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, text: str, k: int = 10) -> list[dict]:
        """Return top-k chunks by BM25 score."""
        if not self._metadata:
            return []

        q_tokens = list(set(_tokenize(text)))
        if not q_tokens:
            return []

        scores: list[float] = [0.0] * self._n

        for term in q_tokens:
            idf = self._idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            for i, tf_dict in enumerate(self._tf):
                f = tf_dict.get(term, 0)
                if f == 0:
                    continue
                dl = self._dl[i]
                norm = _K1 * (1.0 - _B + _B * dl / self._avg_dl)
                scores[i] += idf * f * (_K1 + 1.0) / (f + norm)

        ranked = sorted(
            ((s, i) for i, s in enumerate(scores) if s > 0.0),
            reverse=True,
        )[:k]

        return [
            {**self._metadata[i], "bm25_score": s}
            for s, i in ranked
        ]
