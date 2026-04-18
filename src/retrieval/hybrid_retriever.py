"""Hybrid retriever: 3-way Reciprocal Rank Fusion over dense, sparse, and graph.

Architecture (RAG-Fusion, Rackauckas arXiv:2402.03367 + Edge et al. 2024):

  final_score(d) = Σ_i  weight_i / (RRF_K + rank_i(d))

Three retrievers run in parallel:
  1. Dense vector  — semantic similarity via FAISS + sentence embeddings
  2. Sparse BM25   — exact keyword / section-number recall
  3. Graph BFS     — structural traversal over cross-reference topology

RRF is rank-based, so it needs no score normalisation and is stable across
runs (unlike min-max blending, which is sensitive to outlier scores).

Adaptive per-query weights (tuned on domain characteristics):
  Has § reference  → graph=2.5, dense=1.0, sparse=2.0
    Rationale: graph BFS from that exact node is most precise;
               sparse BM25 also finds the literal section number.
  Numeric query    → graph=1.5, dense=1.5, sparse=1.0
    Rationale: dense embeddings find rate-table text; graph finds
               the structural context; BM25 less discriminative for numbers.
  Default/broad    → graph=2.0, dense=1.0, sparse=1.5
    Rationale: graph community detection handles broad queries well;
               sparse handles any keyword anchors in the query.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

from src import config as cfg
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.utils.ref_patterns import USC_SECTION_RE as _SECTION_REF_RE

_NUMERIC_RE = re.compile(
    r"\b(how much|amount|calculate|total|compute|dollar|percent|rate|bracket"
    r"|threshold|limit|maximum|minimum|\d[\d,]*)\b",
    re.IGNORECASE,
)


class HybridRetriever:
    def __init__(
        self,
        vector: VectorRetriever,
        graph: GraphRetriever,
        sparse: SparseRetriever | None = None,
    ):
        self.vector = vector
        self.graph  = graph
        self.sparse = sparse
        self._query_cache: OrderedDict[tuple[str, str, int, int], list[dict]] = OrderedDict()
        self._query_cache_size = max(0, int(cfg.HYBRID_QUERY_CACHE_SIZE))

    @classmethod
    def load(cls, config) -> "HybridRetriever":
        vector = VectorRetriever.load(
            config.VECTOR_CONTENT_INDEX,
            config.VECTOR_SECTIONID_INDEX,
            config.VECTOR_META_FILE,
            config.EMBEDDING_MODEL,
        )
        graph  = GraphRetriever.load(config.GRAPH_FILE, config.COMMUNITY_FILE)
        sparse = SparseRetriever.build_from_metadata(vector.index.metadata)
        return cls(vector, graph, sparse)

    def query(
        self,
        text: str,
        k: int = 10,
        depth: int = 2,
        mode: str = "hybrid",
    ) -> list[dict]:
        """Return ranked results for one of four retrieval modes.

        Modes:
          "none"   — no retrieval; LLM uses only parametric knowledge
          "vector" — dense FAISS only
          "graph"  — graph BFS / community only
          "hybrid" — 3-way RRF: dense + BM25 + graph
        """
        if mode == "none":
            return []

        cache_key = (mode, text, int(k), int(depth))
        cached = self._cache_get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

        vector_results = self.vector.query(text, k)   if mode in ("vector", "hybrid") else []
        graph_results  = self.graph.query(text, depth) if mode in ("graph",  "hybrid") else []
        sparse_results = (
            self.sparse.query(text, k)
            if mode == "hybrid" and self.sparse is not None
            else []
        )

        if mode == "hybrid":
            dense_w, sparse_w, graph_w = _choose_weights(text)
            merged = _merge_rrf(
                vector_results, sparse_results, graph_results,
                k=k,
                dense_weight=dense_w,
                sparse_weight=sparse_w,
                graph_weight=graph_w,
                rrf_k=cfg.HYBRID_RRF_K,
            )
        elif mode == "vector":
            merged = _rank_with_score(vector_results, k, "vector_score")
        else:
            merged = _rank_with_score(graph_results, k, "graph_score")

        self._cache_set(cache_key, merged)
        return [dict(item) for item in merged]

    # ------------------------------------------------------------------
    # LRU cache
    # ------------------------------------------------------------------

    def _cache_get(self, key: tuple) -> list[dict] | None:
        if self._query_cache_size <= 0:
            return None
        cached = self._query_cache.get(key)
        if cached is None:
            return None
        self._query_cache.move_to_end(key)
        return list(cached)

    def _cache_set(self, key: tuple, value: list[dict]) -> None:
        if self._query_cache_size <= 0:
            return
        self._query_cache[key] = list(value)
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Adaptive weight selection
# ---------------------------------------------------------------------------

def _choose_weights(text: str) -> tuple[float, float, float]:
    """Return (dense_weight, sparse_weight, graph_weight) for this query.

    Graph is consistently stronger in this domain (mrr_hier 0.47 vs 0.07 for
    dense in a large tax corpus), so graph weight is always >= dense weight.
    Sparse BM25 earns a higher weight when exact keyword recall matters most.
    """
    has_section_ref = bool(_SECTION_REF_RE.search(text))
    is_numeric      = bool(_NUMERIC_RE.search(text))

    if has_section_ref:
        # Graph BFS from that node is highly precise; sparse also finds the
        # literal section number text; dense is least helpful here.
        return 1.0, 2.0, 2.5
    if is_numeric:
        # Dense finds rate-table text; graph finds definitions/structure;
        # BM25 is less helpful for numeric queries that lack section anchors.
        return 1.5, 1.0, 1.5
    # Default / broad conceptual: graph community detection + BM25 keyword
    # anchors complement each other; dense adds semantic breadth.
    return 1.0, 1.5, 2.0


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _merge_rrf(
    vector_results: list[dict],
    sparse_results: list[dict],
    graph_results:  list[dict],
    k: int,
    dense_weight:  float = 1.0,
    sparse_weight: float = 1.5,
    graph_weight:  float = 2.0,
    rrf_k: int = 60,
) -> list[dict]:
    """3-way Reciprocal Rank Fusion.

    score(d) = dense_weight  / (rrf_k + rank_dense(d))
             + sparse_weight / (rrf_k + rank_sparse(d))
             + graph_weight  / (rrf_k + rank_graph(d))

    Chunks absent from a list contribute 0 from that list (rank = ∞).
    Chunks appearing in multiple lists are naturally boosted — the standard
    RRF 'voting' property that rewards cross-retriever agreement.
    """
    merged: dict[str, dict] = {}

    def _add(results: list[dict], weight: float, score_key: str) -> None:
        for rank, r in enumerate(results, 1):
            sid = r["section_id"]
            if sid not in merged:
                merged[sid] = {
                    **r,
                    "vector_score": 0.0,
                    "bm25_score":   0.0,
                    "graph_score":  0.0,
                    "rrf_score":    0.0,
                }
            # Keep best raw score per channel for debugging / inspection.
            raw = float(r.get(score_key, 0.0))
            if raw > merged[sid].get(score_key, 0.0):
                merged[sid][score_key] = raw
                if "text" in r and len(r["text"]) > len(merged[sid].get("text", "")):
                    merged[sid]["text"] = r["text"]
            merged[sid]["rrf_score"] += weight / (rrf_k + rank)

    _add(vector_results, dense_weight,  "vector_score")
    _add(sparse_results, sparse_weight, "bm25_score")
    _add(graph_results,  graph_weight,  "graph_score")

    for r in merged.values():
        r["score"] = r["rrf_score"]

    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:k]


def _rank_with_score(results: list[dict], k: int, score_key: str) -> list[dict]:
    """Standardise single-retriever output with a `score` field."""
    out = []
    for r in results[:k]:
        item = dict(r)
        item["score"] = float(item.get(score_key, 0.0))
        out.append(item)
    return out
