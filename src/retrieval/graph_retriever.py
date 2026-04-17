"""GraphRAG retriever.

Two retrieval strategies are combined, following Edge et al. (2024):

  1. Community-level retrieval (broad queries):
     When the query contains no explicit section reference and is short/general,
     communities are ranked by keyword overlap and their summaries are returned.

  2. Node-level retrieval (specific queries):
     Entry nodes are matched by explicit § references plus topic hints, then
     expanded by multi-hop graph traversal over hierarchy/xref/coverage edges.

This module is intentionally section-aware so it can recover chains such as:
  Pub 17 -> Pub 596 -> Schedule EIC -> 26 USC §32
"""

import re
from collections import OrderedDict
from pathlib import Path

from src import config as cfg
from src.indexing.graph_index import GraphIndex
from src.utils.ref_patterns import (
    USC_SECTION_RE as _SECTION_REF_RE,
    USC_SECTION_RANGE_RE as _SECTION_RANGE_RE,
    PUB_RE as _PUB_REF_RE,
    FORM_RE as _FORM_RE,
    SCHEDULE_RE as _SCHEDULE_RE,
    FORM_SCHEDULE_RE as _FORM_SCHEDULE_RE,
)


# A query is considered "broad" if it has no section reference and
# fewer than this many words.
BROAD_QUERY_WORD_LIMIT = 12

# Number of top communities to return for broad queries.
TOP_COMMUNITIES = 3

# Soft cap to avoid flooding results with weak title matches.
MAX_ENTRY_NODES = max(1, cfg.GRAPH_MAX_ENTRY_NODES)
MAX_NEIGHBORS_PER_NODE = max(1, cfg.GRAPH_MAX_NEIGHBORS_PER_NODE)

# Basic stop-word list for query tokenization.
STOPWORDS = {
    "about", "after", "also", "and", "any", "are", "can", "for", "from",
    "how", "into", "its", "like", "line", "lines", "more", "that", "the",
    "their", "them", "then", "there", "these", "this", "those", "what",
    "when", "where", "which", "with", "would", "your",
}

FORM_SCHEDULE_PREFIXES: dict[str, str] = {
    "schedule b": "sch. b instructions",
    "schedule c": "sch. c instructions",
    "schedule d": "sch. d instructions",
    "schedule e": "sch. e instructions",
    "schedule eic": "irs pub. 596",
    "schedule se": "sch. se instructions",
    "schedule 8812": "sch. 8812 instructions",
    "form 1040": "form 1040 instructions",
    "form 1040-es": "form 1040-es instructions",
    "form 1099-q": "form 1099-q instructions",
    "form 2441": "form 2441 instructions",
    "form 4562": "form 4562 instructions",
    "form 8606": "form 8606 instructions",
    "form 8829": "form 8829 instructions",
    "form 8863": "form 8863 instructions",
    "form 8949": "form 8949 instructions",
    "form 8995": "form 8995 instructions",
}

# Domain hints from user wording -> likely IRC entry sections.
# These boost discovery of sections that are not lexically similar to query
# terms (e.g. "child" doesn't appear in the text of §152 headers).
KEYWORD_TO_SECTION_HINTS: dict[str, set[str]] = {
    "agi":        {"26 usc §62"},
    "adjusted":   {"26 usc §62"},
    "aoc":        {"26 usc §25a"},
    "annuities":  {"26 usc §72"},
    "annuity":    {"26 usc §72"},
    "capital":    {"26 usc §1001", "26 usc §1211", "26 usc §1212", "26 usc §1221",
                   "26 usc §1222", "26 usc §1223"},
    "care":       {"26 usc §21"},
    "charitable": {"26 usc §170"},
    "child":      {"26 usc §21", "26 usc §24", "26 usc §152"},
    "credit":     {"26 usc §21", "26 usc §22", "26 usc §24", "26 usc §25a", "26 usc §32"},
    "deduction":  {"26 usc §63", "26 usc §163", "26 usc §164", "26 usc §170",
                   "26 usc §213", "26 usc §219"},
    "dependent":  {"26 usc §21", "26 usc §24", "26 usc §152"},
    "disabled":   {"26 usc §22"},
    "eic":        {"26 usc §32"},
    "earned":     {"26 usc §32"},
    "education":  {"26 usc §25a"},
    "elderly":    {"26 usc §22"},
    "exclusion":  {"26 usc §101", "26 usc §102", "26 usc §104", "26 usc §105",
                   "26 usc §108", "26 usc §121"},
    "gain":       {"26 usc §1001", "26 usc §1222"},
    "gross":      {"26 usc §61"},
    "home":       {"26 usc §121", "26 usc §163"},
    "income":     {"26 usc §61", "26 usc §62", "26 usc §63"},
    "interest":   {"26 usc §163"},
    "ira":        {"26 usc §72", "26 usc §219", "26 usc §408"},
    "llc":        {"26 usc §25a"},
    "loss":       {"26 usc §165", "26 usc §1001", "26 usc §1211", "26 usc §1212"},
    "medical":    {"26 usc §213"},
    "mortgage":   {"26 usc §163"},
    "passive":    {"26 usc §469"},
    "pease":      {"26 usc §68"},
    "retirement": {"26 usc §72", "26 usc §401", "26 usc §408"},
    "sale":       {"26 usc §121", "26 usc §1001"},
    "salt":       {"26 usc §164"},
    "security":   {"26 usc §86"},
    "social":     {"26 usc §86"},
    "standard":   {"26 usc §63"},
    "taxable":    {"26 usc §63", "26 usc §86"},
}


class GraphRetriever:
    def __init__(self, index: GraphIndex):
        self.index = index
        self._query_cache: OrderedDict[tuple[str, int], list[dict]] = OrderedDict()
        self._cache_size = max(0, int(cfg.GRAPH_QUERY_CACHE_SIZE))

        self._node_text: dict[str, tuple[str, str, str]] = {}
        for node_id, attrs in self.index.graph.nodes(data=True):
            self._node_text[node_id] = (
                str(node_id).lower(),
                str(attrs.get("title", "")).lower(),
                str(attrs.get("source", "")).lower(),
            )

    @classmethod
    def load(cls, graph_path: Path, community_path: Path):
        index = GraphIndex()
        index.load(graph_path, community_path)
        return cls(index)

    def query(self, text: str, depth: int = 2) -> list[dict]:
        """Return ranked results using community and/or node-level retrieval.

        Results include a `graph_score` field. Community summary results are
        scored 0.9; node-level scores decay with hop distance.
        """
        depth = max(0, int(depth))
        cache_key = (text, depth)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

        has_section_ref = bool(_SECTION_REF_RE.search(text))
        word_count = len(text.split())
        is_broad = not has_section_ref and word_count <= BROAD_QUERY_WORD_LIMIT

        results: dict[str, dict] = {}

        if is_broad:
            for community_result in self._query_communities(text):
                sid = community_result["section_id"]
                results[sid] = community_result

        # Always also run node-level retrieval and merge.
        for node_result in self._query_nodes(text, depth):
            sid = node_result["section_id"]
            if sid not in results or node_result["graph_score"] > results[sid]["graph_score"]:
                results[sid] = node_result

        ranked = sorted(results.values(), key=lambda r: r["graph_score"], reverse=True)
        self._cache_set(cache_key, ranked)
        return [dict(item) for item in ranked]

    def _query_communities(self, text: str) -> list[dict]:
        """Return community summary results ranked by keyword overlap with query."""
        query_words = set(_query_terms(text))

        scored = []
        for community_id, community in self.index.communities.items():
            community_words = set(community.get("keywords", []))
            overlap = len(query_words & community_words)
            if overlap > 0:
                scored.append((overlap, community_id, community))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:TOP_COMMUNITIES]

        results = []
        for _, community_id, community in top:
            results.append({
                "section_id": f"community:{community_id}",
                "title":      f"Community {community_id} ({community['size']} sections)",
                "source":     "graph_community",
                "text":       community["summary"],
                "snippet":    community["summary"][:300],
                "graph_score": 0.9,
            })
        return results

    def _query_nodes(self, text: str, depth: int) -> list[dict]:
        """Find entry nodes and expand by multi-hop traversal.

        Traversal is frontier-based (not repeated one-hop expansion) and ranks
        neighbors by edge type and query overlap.
        """
        entry_matches = self._find_entry_nodes(text)
        if not entry_matches:
            return []

        max_depth = max(0, depth)
        query_terms = _query_terms(text)
        entry_score_map = dict(entry_matches)

        # Entry node score + per-hop decay.
        base_scores = [0.90, 0.78, 0.66, 0.56, 0.48]

        results: dict[str, dict] = {}
        visited: set[str] = set()
        frontier: set[str] = set(entry_score_map)

        for hop in range(max_depth + 1):
            if not frontier:
                break

            if hop < len(base_scores):
                hop_score = base_scores[hop]
            else:
                hop_score = max(0.30, base_scores[-1] - 0.08 * (hop - len(base_scores) + 1))

            next_frontier: set[str] = set()

            for node_id in sorted(frontier):
                if node_id not in self.index.graph or node_id in visited:
                    continue

                visited.add(node_id)
                node_data = dict(self.index.graph.nodes[node_id])

                node_score = hop_score
                if hop == 0:
                    node_score = entry_score_map.get(node_id, hop_score)

                existing = results.get(node_id)
                if existing is None or node_score > existing["graph_score"]:
                    results[node_id] = {
                        "section_id": node_id,
                        "graph_score": node_score,
                        **node_data,
                    }

                if hop >= max_depth:
                    continue

                next_frontier.update(self._ordered_neighbors(node_id, query_terms))

            frontier = next_frontier - visited

        return list(results.values())

    def _ordered_neighbors(self, node_id: str, query_terms: list[str]) -> list[str]:
        """Return neighbors ordered by edge informativeness and query overlap."""
        scored: list[tuple[float, str]] = []
        seen: set[str] = set()

        for neighbor in self.index.graph.successors(node_id):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            edge_attrs = dict(self.index.graph.edges[node_id, neighbor])
            edge_type = edge_attrs.get("type", "")
            score = self._neighbor_priority(neighbor, edge_type, query_terms, edge_attrs)
            scored.append((score, neighbor))

        for neighbor in self.index.graph.predecessors(node_id):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            edge_attrs = dict(self.index.graph.edges[neighbor, node_id])
            edge_type = edge_attrs.get("type", "")
            score = self._neighbor_priority(neighbor, edge_type, query_terms, edge_attrs)
            scored.append((score, neighbor))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [node for _, node in scored[:MAX_NEIGHBORS_PER_NODE]]

    def _neighbor_priority(
        self,
        neighbor_id: str,
        edge_type: str,
        query_terms: list[str],
        edge_attrs: dict | None = None,
    ) -> float:
        """Score a neighbor candidate for BFS frontier ordering."""
        edge_weight = {
            "xref":      cfg.GRAPH_EDGE_WEIGHT_XREF,
            "coverage":  cfg.GRAPH_EDGE_WEIGHT_COVERAGE,
            "hierarchy": cfg.GRAPH_EDGE_WEIGHT_HIERARCHY,
        }.get(edge_type, cfg.GRAPH_EDGE_WEIGHT_DEFAULT)

        if edge_type == "coverage":
            provenance = str((edge_attrs or {}).get("coverage_provenance", "")).lower()
            if provenance == "fallback":
                edge_weight -= cfg.GRAPH_COVERAGE_PENALTY_FALLBACK
            elif provenance.startswith("inferred"):
                edge_weight -= cfg.GRAPH_COVERAGE_PENALTY_INFERRED

        attrs = self.index.graph.nodes[neighbor_id]
        text = f"{neighbor_id} {attrs.get('title', '')}".lower()
        overlap = sum(1 for term in query_terms if term in text)
        usc_boost = cfg.GRAPH_USC_BOOST if attrs.get("source") == "usc26" else 0.0

        return max(0.0, edge_weight + cfg.GRAPH_OVERLAP_WEIGHT * overlap + usc_boost)

    def _find_entry_nodes(self, text: str) -> list[tuple[str, float]]:
        """Return likely entry nodes with normalized confidence scores."""
        text_lower = text.lower()
        query_terms = _query_terms(text)
        section_refs = _extract_section_refs(text)
        irs_prefix_refs = _extract_irs_prefix_refs(text)
        section_ref_candidates: set[str] = set()

        hinted_prefixes: set[str] = set()
        for term in query_terms:
            hinted_prefixes.update(KEYWORD_TO_SECTION_HINTS.get(term, set()))

        scored_matches: dict[str, float] = {}

        for node_id, (node_id_lower, title, source_lower) in self._node_text.items():
            score = 0.0
            overlap = sum(1 for term in query_terms if term in node_id_lower or term in title)

            section_ref_match = any(
                _section_ref_matches_node(section_ref, node_id_lower)
                for section_ref in section_refs
            )
            if section_ref_match:
                score += 6.0
                section_ref_candidates.add(node_id)

            if irs_prefix_refs and any(node_id_lower.startswith(prefix) for prefix in irs_prefix_refs):
                score += 2.5
                score += min(2.0, 0.5 * overlap)

            if hinted_prefixes and any(node_id_lower.startswith(prefix) for prefix in hinted_prefixes):
                score += 4.0

            if overlap > 0:
                score += min(3.0, float(overlap))

            # If no extracted terms but query is short, allow title substring match.
            if score == 0.0 and not query_terms and text_lower in title:
                score += 0.5

            if score > 0.0:
                if source_lower == "usc26":
                    score += 0.1
                scored_matches[node_id] = score

        if not scored_matches:
            return []

        if section_refs and section_ref_candidates:
            scored_matches = {
                node_id: score
                for node_id, score in scored_matches.items()
                if node_id in section_ref_candidates
            }
            if not scored_matches:
                return []

        max_raw = max(scored_matches.values())

        threshold = max(cfg.GRAPH_ENTRY_THRESHOLD_BASE, max_raw - 2.0)
        if section_refs:
            threshold = max(threshold, cfg.GRAPH_ENTRY_THRESHOLD_SECTION)

        filtered = [
            (node_id, raw)
            for node_id, raw in scored_matches.items()
            if raw >= threshold
        ]
        if not filtered:
            filtered = sorted(
                scored_matches.items(),
                key=lambda item: (-item[1], item[0]),
            )[:10]

        ranked = sorted(filtered, key=lambda item: (-item[1], item[0]))[:MAX_ENTRY_NODES]

        # Map raw confidence to graph score range [0.60, 1.00].
        normalized: list[tuple[str, float]] = []
        for node_id, raw in ranked:
            confidence = 0.60 + 0.40 * (raw / max_raw)
            normalized.append((node_id, min(1.0, confidence)))

        return normalized

    def _cache_get(self, key: tuple[str, int]) -> list[dict] | None:
        if self._cache_size <= 0:
            return None

        cached = self._query_cache.get(key)
        if cached is None:
            return None

        self._query_cache.move_to_end(key)
        return list(cached)

    def _cache_set(self, key: tuple[str, int], results: list[dict]) -> None:
        if self._cache_size <= 0:
            return

        self._query_cache[key] = list(results)
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > self._cache_size:
            self._query_cache.popitem(last=False)


def _query_terms(text: str) -> list[str]:
    """Tokenize query text into lowercase content words."""
    terms = []
    for word in re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower()):
        if word not in STOPWORDS:
            terms.append(word)
    return terms


def _extract_section_refs(text: str) -> set[str]:
    """Extract canonical lowercase 26 USC section references from query text."""
    refs: set[str] = set()

    # Expand section ranges like §101-108.
    for start_raw, end_raw in _SECTION_RANGE_RE.findall(text):
        start = int(start_raw)
        end = int(end_raw)
        if start <= end and (end - start) <= 60:
            for value in range(start, end + 1):
                refs.add(f"26 usc §{value}")

    # Extract simple and subsection references.
    for num, sub in _SECTION_REF_RE.findall(text):
        num_norm = num.lower()
        if sub:
            refs.add(f"26 usc §{num_norm}{sub.lower()}")
        refs.add(f"26 usc §{num_norm}")

    return refs


def _section_ref_matches_node(section_ref: str, node_id_lower: str) -> bool:
    """Return True when a query section ref matches a graph node section id.

    Supports exact matches and descendant subsection prefix matches, e.g.:
      ref:  26 usc §151
      node: 26 usc §151(d)
    """
    if node_id_lower == section_ref:
        return True
    return node_id_lower.startswith(section_ref + "(")


def _extract_irs_prefix_refs(text: str) -> set[str]:
    """Extract lowercase IRS source-id prefixes from publication/form mentions."""
    refs: set[str] = set()

    for pub_num in _PUB_REF_RE.findall(text):
        refs.add(f"irs pub. {pub_num}")

    for raw in _FORM_SCHEDULE_RE.findall(text):
        key = re.sub(r"\s+", " ", raw.lower().strip())
        mapped = FORM_SCHEDULE_PREFIXES.get(key)
        if mapped:
            refs.add(mapped)

    return refs
