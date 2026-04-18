"""Hybrid retriever: merges vector and graph results via Reciprocal Rank Fusion.

Following Edge et al. (2024), both retrievers run independently and results are
fused using Reciprocal Rank Fusion (RRF):

    rrf_score(d) = Σ_i  weight_i / (k + rank_i(d))

where k=60 (standard constant) and weight_i is a per-query adaptive weight.
RRF is rank-based, so it never requires score normalization and is insensitive
to the raw score distributions of each retriever.  Chunks appearing in both
lists naturally score ~2× a chunk appearing in only one list.

Adaptive weights (graph is consistently better in this domain):
  - Query has explicit § reference  → graph_weight=2.5, vector_weight=1.0
  - Query is numeric/calculation     → graph_weight=1.5, vector_weight=1.5
  - Query is broad/conceptual        → graph_weight=2.0, vector_weight=1.0
  - Default                          → graph_weight=2.0, vector_weight=1.0
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

from src import config as cfg
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.utils.ref_patterns import USC_SECTION_RE as _SECTION_REF_RE

_NUMERIC_RE = re.compile(
    r"\b(how much|amount|calculate|total|compute|dollar|percent|rate|bracket"
    r"|threshold|limit|maximum|minimum|\d[\d,]*)\b",
    re.IGNORECASE,
)
_BROAD_RE = re.compile(
    r"^(what is|what are|define|explain|describe|overview|generally|in general)\b",
    re.IGNORECASE,
)


class HybridRetriever:
    def __init__(self, vector: VectorRetriever, graph: GraphRetriever):
        self.vector = vector
        self.graph  = graph
        self._query_cache: OrderedDict[tuple[str, str, int, int], list[dict]] = OrderedDict()
        self._query_cache_size = max(0, int(cfg.HYBRID_QUERY_CACHE_SIZE))

    @classmethod
    def load(cls, config):
        vector = VectorRetriever.load(
            config.VECTOR_CONTENT_INDEX,
            config.VECTOR_SECTIONID_INDEX,
            config.VECTOR_META_FILE,
            config.EMBEDDING_MODEL,
        )
        graph = GraphRetriever.load(config.GRAPH_FILE, config.COMMUNITY_FILE)
        return cls(vector, graph)

    def query(
        self,
        text: str,
        k: int = 10,
        depth: int = 2,
        mode: str = "hybrid",
    ) -> list[dict]:
        """Return ranked results for one of four experimental conditions.

        mode options:
          "none"   - no retrieval (returns empty list)
          "vector" - vector index only
          "graph"  - graph index only
          "hybrid" - RRF fusion of both indexes with adaptive per-query weights
        """
        if mode == "none":
            return []

        cache_key = (mode, text, int(k), int(depth))
        cached = self._cache_get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

        vector_results = self.vector.query(text, k) if mode in ("vector", "hybrid") else []
        graph_results  = self.graph.query(text, depth) if mode in ("graph",  "hybrid") else []

        if mode == "hybrid":
            vector_weight, graph_weight = _choose_weights(text)
            merged = _merge_rrf(
                vector_results, graph_results, k, vector_weight, graph_weight,
                rrf_k=cfg.HYBRID_RRF_K,
            )
        elif mode == "vector":
            merged = _rank_with_score(vector_results, k, score_key="vector_score")
        else:
            merged = _rank_with_score(graph_results, k, score_key="graph_score")

        self._cache_set(cache_key, merged)
        return [dict(item) for item in merged]

    def _cache_get(self, key: tuple[str, str, int, int]) -> list[dict] | None:
        if self._query_cache_size <= 0:
            return None
        cached = self._query_cache.get(key)
        if cached is None:
            return None
        self._query_cache.move_to_end(key)
        return list(cached)

    def _cache_set(self, key: tuple[str, str, int, int], value: list[dict]) -> None:
        if self._query_cache_size <= 0:
            return
        self._query_cache[key] = list(value)
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)


def _choose_weights(text: str) -> tuple[float, float]:
    """Return (vector_weight, graph_weight) for RRF based on query characteristics.

    Graph retrieval is stronger in this domain (mrr_hier 0.47 vs 0.07 for vector
    in the full 2017 corpus), so graph weight is always >= vector weight.
    The only exception is numeric/calculation queries where vector similarity
    finds rate tables and bracket text reliably.
    """
    has_section_ref = bool(_SECTION_REF_RE.search(text))
    is_numeric      = bool(_NUMERIC_RE.search(text))

    if has_section_ref:
        # Explicit § reference: graph BFS from that node is highly precise.
        return 1.0, 2.5
    if is_numeric and not has_section_ref:
        # Calculation questions: vector finds number tables; graph finds definitions.
        return 1.5, 1.5
    # Default / broad conceptual: graph community detection is more reliable.
    return 1.0, 2.0


def _merge_rrf(
    vector_results: list[dict],
    graph_results: list[dict],
    k: int,
    vector_weight: float = 1.0,
    graph_weight: float = 1.0,
    rrf_k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion over two ranked lists.

    score(d) = vector_weight/(rrf_k + rank_v(d)) + graph_weight/(rrf_k + rank_g(d))

    Chunks missing from a list contribute 0 from that list (i.e. they are
    treated as ranked at infinity).  This avoids inflating scores for chunks
    that only one retriever found.
    """
    merged: dict[str, dict] = {}

    for rank, r in enumerate(vector_results, 1):
        sid = r["section_id"]
        if sid not in merged:
            merged[sid] = {**r, "vector_score": r.get("vector_score", 0.0),
                           "graph_score": 0.0, "rrf_score": 0.0}
        merged[sid]["rrf_score"] += vector_weight / (rrf_k + rank)

    for rank, r in enumerate(graph_results, 1):
        sid = r["section_id"]
        if sid in merged:
            merged[sid]["graph_score"] = max(
                merged[sid]["graph_score"], r.get("graph_score", 0.0)
            )
            if "text" in r and len(r["text"]) > len(merged[sid].get("text", "")):
                merged[sid]["text"] = r["text"]
        else:
            merged[sid] = {**r, "vector_score": 0.0,
                           "graph_score": r.get("graph_score", 0.0), "rrf_score": 0.0}
        merged[sid]["rrf_score"] += graph_weight / (rrf_k + rank)

    for r in merged.values():
        r["score"] = r["rrf_score"]

    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:k]


def _rank_with_score(results: list[dict], k: int, score_key: str) -> list[dict]:
    """Standardise single-retriever results to have a `score` field."""
    out = []
    for r in results[:k]:
        item = dict(r)
        item["score"] = float(item.get(score_key, 0.0))
        out.append(item)
    return out
