"""Hybrid retriever: merges vector and graph results, deduplicates, and re-ranks.

Scoring:
  final_score = alpha * vector_score + (1 - alpha) * graph_score

Alpha is adaptive:
  - Default (broad/semantic query): HYBRID_ALPHA_DEFAULT = 0.6  (vector-weighted)
  - Explicit § reference in query:  HYBRID_ALPHA_SECTION_REF = 0.35 (graph-weighted)

This means queries like "what is the standard deduction?" lean on vector similarity,
while queries like "what does §32(c) say about investment income?" lean on the
graph to follow the exact section and its neighbors.

Deduplication is by section_id. When the same section appears in both
retrieval paths, scores are combined rather than the result being dropped.
"""

from pathlib import Path

from src import config as cfg
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.utils.ref_patterns import USC_SECTION_RE as _SECTION_REF_RE


class HybridRetriever:
    def __init__(self, vector: VectorRetriever, graph: GraphRetriever):
        self.vector = vector
        self.graph  = graph

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

        vector_results = self.vector.query(text, k) if mode in ("vector", "hybrid") else []
        graph_results  = self.graph.query(text, depth) if mode in ("graph",  "hybrid") else []

        alpha = _choose_alpha(text)
        return self._merge(vector_results, graph_results, k, alpha)

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
            merged[sid] = {**r, "vector_score": r.get("vector_score", 0.0), "graph_score": 0.0}

        for r in graph_results:
            sid = r["section_id"]
            if sid in merged:
                merged[sid]["graph_score"] = r.get("graph_score", 0.0)
            else:
                merged[sid] = {**r, "vector_score": 0.0, "graph_score": r.get("graph_score", 0.0)}

        for r in merged.values():
            r["score"] = alpha * r["vector_score"] + (1 - alpha) * r["graph_score"]

        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:k]


def _choose_alpha(text: str) -> float:
    """Return the blending alpha appropriate for this query.

    If the query contains an explicit IRC § reference (e.g. "§32", "section 163"),
    the graph index is more likely to return the exact node and its neighbors, so
    we down-weight vector similarity in favor of graph scores.
    """
    if _SECTION_REF_RE.search(text):
        return cfg.HYBRID_ALPHA_SECTION_REF
    return cfg.HYBRID_ALPHA_DEFAULT
