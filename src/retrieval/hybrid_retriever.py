"""Hybrid retriever: merges vector and graph results, deduplicates, and re-ranks.

Scoring:
  final_score = alpha * vector_score + (1 - alpha) * graph_score

Alpha is adaptive:
  - Default (broad/semantic query): HYBRID_ALPHA_DEFAULT = 0.6  (vector-weighted)
    - Explicit § reference in query:  HYBRID_ALPHA_SECTION_REF = 0.5  (more balanced)

This means queries like "what is the standard deduction?" lean on vector similarity,
while queries like "what does §32(c) say about investment income?" lean on the
graph to follow the exact section and its neighbors.

Deduplication is by section_id. When the same section appears in both
retrieval paths, scores are combined rather than the result being dropped.
"""

from pathlib import Path
from collections import OrderedDict

from src import config as cfg
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.utils.ref_patterns import USC_SECTION_RE as _SECTION_REF_RE


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
          "none"   - no retrieval (returns empty list, LLM uses no context)
          "vector" - vector index only
          "graph"  - graph index only
          "hybrid" - both indexes merged with adaptive alpha blending
        """
        if mode == "none":
            return []

        cache_key = (mode, text, int(k), int(depth))
        cached = self._cache_get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

        vector_results = self.vector.query(text, k) if mode in ("vector", "hybrid") else []
        graph_results  = self.graph.query(text, depth) if mode in ("graph",  "hybrid") else []

        alpha = _choose_alpha(text)
        merged = self._merge(vector_results, graph_results, k, alpha)
        self._cache_set(cache_key, merged)
        return [dict(item) for item in merged]

    def _merge(
        self,
        vector_results: list[dict],
        graph_results:  list[dict],
        k: int,
        alpha: float,
    ) -> list[dict]:
        merged = {}

        for r in vector_results:
            sid = r["section_id"]
            if sid not in merged:
                merged[sid] = {**r, "vector_score": r.get("vector_score", 0.0), "graph_score": 0.0}

        for r in graph_results:
            sid = r["section_id"]
            if sid in merged:
                merged[sid]["graph_score"] = max(merged[sid]["graph_score"], r.get("graph_score", 0.0))
                if "text" in r and len(r["text"]) > len(merged[sid].get("text", "")):
                    merged[sid]["text"] = r["text"]
            else:
                merged[sid] = {**r, "vector_score": 0.0, "graph_score": r.get("graph_score", 0.0)}

        if cfg.HYBRID_SCORE_NORMALIZE:
            vector_norm = _minmax_normalize(
                {sid: float(r.get("vector_score", 0.0)) for sid, r in merged.items()}
            )
            graph_norm = _minmax_normalize(
                {sid: float(r.get("graph_score", 0.0)) for sid, r in merged.items()}
            )
        else:
            vector_norm = {sid: float(r.get("vector_score", 0.0)) for sid, r in merged.items()}
            graph_norm = {sid: float(r.get("graph_score", 0.0)) for sid, r in merged.items()}

        for r in merged.values():
            sid = r["section_id"]
            r["vector_score_norm"] = vector_norm.get(sid, 0.0)
            r["graph_score_norm"] = graph_norm.get(sid, 0.0)
            r["score"] = alpha * r["vector_score_norm"] + (1 - alpha) * r["graph_score_norm"]

        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:k]

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


def _choose_alpha(text: str) -> float:
    """Return the blending alpha appropriate for this query.

    If the query contains an explicit IRC § reference (e.g. "§32", "section 163"),
    the graph index is more likely to return the exact node and its neighbors, so
    we down-weight vector similarity in favor of graph scores.
    """
    if _SECTION_REF_RE.search(text):
        return cfg.HYBRID_ALPHA_SECTION_REF
    return cfg.HYBRID_ALPHA_DEFAULT


def _minmax_normalize(values: dict[str, float]) -> dict[str, float]:
    """Normalize values to [0, 1] for stable score blending."""
    if not values:
        return {}

    lo = min(values.values())
    hi = max(values.values())
    if hi - lo <= 1e-9:
        return {key: 1.0 for key in values}

    span = hi - lo
    return {key: (value - lo) / span for key, value in values.items()}
