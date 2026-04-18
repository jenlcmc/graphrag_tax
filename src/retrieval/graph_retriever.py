"""GraphRAG retriever.

Two retrieval strategies are combined, following Edge et al. (2024):

  1. Community-level retrieval (broad queries):
     Communities are ranked by TF-IDF-weighted keyword overlap with the query.
     IDF is computed across all communities at init time so rare discriminative
     terms (e.g. "passthrough") outweigh common ones (e.g. "income").

  2. Node-level retrieval (specific queries):
     Entry nodes are matched by explicit § references plus topic hints, then
     expanded by multi-hop graph traversal over hierarchy/xref/coverage edges.
     Falls back to community mode when no entry nodes are found.
"""

import math
import re
from collections import OrderedDict
from pathlib import Path

from src import config as cfg
from src.indexing.graph_index import GraphIndex
from src.utils.ref_patterns import (
    USC_SECTION_RE as _SECTION_REF_RE,
    USC_SECTION_RANGE_RE as _SECTION_RANGE_RE,
    PUB_RE as _PUB_REF_RE,
    FORM_SCHEDULE_RE as _FORM_SCHEDULE_RE,
)


BROAD_QUERY_WORD_LIMIT = 12
TOP_COMMUNITIES = 3
MAX_ENTRY_NODES = max(1, cfg.GRAPH_MAX_ENTRY_NODES)
MAX_NEIGHBORS_PER_NODE = max(1, cfg.GRAPH_MAX_NEIGHBORS_PER_NODE)

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

        # Pre-compute lowercased node text for matching.
        self._node_text: dict[str, tuple[str, str, str]] = {}
        for node_id, attrs in self.index.graph.nodes(data=True):
            self._node_text[node_id] = (
                str(node_id).lower(),
                str(attrs.get("title", "")).lower(),
                str(attrs.get("source", "")).lower(),
            )

        # Inverted index: content word -> node_ids.
        # Avoids scanning all nodes per query in _find_entry_nodes.
        self._term_to_nodes: dict[str, set[str]] = {}
        for node_id, (node_id_lower, title, _) in self._node_text.items():
            for word in re.findall(r"\b[a-zA-Z0-9]{2,}\b", node_id_lower + " " + title):
                if word not in STOPWORDS:
                    self._term_to_nodes.setdefault(word, set()).add(node_id)

        # Section ref index: canonical section ref -> all matching node_ids.
        # A ref "26 usc §32" matches node "26 usc §32" and all its subsections.
        # A ref "26 usc §32(c)" matches "26 usc §32(c)" and its sub-subsections.
        self._section_ref_to_nodes: dict[str, set[str]] = {}
        for node_id, (node_id_lower, _, _) in self._node_text.items():
            if "§" not in node_id_lower:
                continue
            # Index node under its own id and every parent section ref.
            current = node_id_lower
            while True:
                self._section_ref_to_nodes.setdefault(current, set()).add(node_id)
                if "(" not in current:
                    break
                current = re.sub(r"\([^()]*\)$", "", current).strip()

        # Non-USC nodes for IRS prefix matching (avoids scanning USC nodes).
        self._irs_nodes: list[tuple[str, str]] = [
            (node_id_lower, node_id)
            for node_id, (node_id_lower, _, _) in self._node_text.items()
            if not node_id_lower.startswith("26 usc §")
        ]

        # IDF over community keywords for TF-IDF weighted community scoring.
        # Rare keywords that appear in few communities are more discriminative.
        n_communities = max(1, len(self.index.communities))
        keyword_df: dict[str, int] = {}
        for community in self.index.communities.values():
            for kw in set(community.get("keywords", [])):
                keyword_df[kw] = keyword_df.get(kw, 0) + 1
        self._community_idf: dict[str, float] = {
            kw: math.log((n_communities + 1) / (df + 1))
            for kw, df in keyword_df.items()
        }

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

        node_results = self._query_nodes(text, depth)
        for node_result in node_results:
            sid = node_result["section_id"]
            if sid not in results or node_result["graph_score"] > results[sid]["graph_score"]:
                results[sid] = node_result

        # Fallback: specific query found no entry nodes → use community mode
        # rather than returning empty-handed.
        if not node_results and not is_broad:
            for community_result in self._query_communities(text):
                sid = community_result["section_id"]
                if sid not in results:
                    results[sid] = community_result

        ranked = sorted(results.values(), key=lambda r: r["graph_score"], reverse=True)
        self._cache_set(cache_key, ranked)
        return [dict(item) for item in ranked]

    def _query_communities(self, text: str) -> list[dict]:
        """Rank communities by TF-IDF-weighted keyword overlap with the query.

        Using IDF means rare discriminative keywords (e.g. "passthrough",
        "exemption") contribute more than common ones (e.g. "income", "tax").
        Raw overlap count treats all matching words equally, which biases toward
        larger communities that happen to contain many common words.
        """
        query_words = set(_query_terms(text))
        if not query_words:
            return []

        scored = []
        for community_id, community in self.index.communities.items():
            community_words = set(community.get("keywords", []))
            matched = query_words & community_words
            if not matched:
                continue
            idf_score = sum(self._community_idf.get(w, 0.0) for w in matched)
            if idf_score > 0.0:
                scored.append((idf_score, community_id, community))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Normalise graph_score: top community = 0.90, others scaled down.
        if not scored:
            return []
        max_score = scored[0][0]
        results = []
        for idf_score, community_id, community in scored[:TOP_COMMUNITIES]:
            normalised = 0.70 + 0.20 * (idf_score / max_score)
            results.append({
                "section_id":  f"community:{community_id}",
                "title":       f"Community {community_id} ({community['size']} sections)",
                "source":      "graph_community",
                "text":        community["summary"],
                "snippet":     community["summary"][:300],
                "graph_score": normalised,
            })
        return results

    def _query_nodes(self, text: str, depth: int) -> list[dict]:
        """Find entry nodes and expand by multi-hop traversal.

        Traversal is frontier-based and ranks neighbors by edge type and query overlap.
        """
        entry_matches = self._find_entry_nodes(text)
        if not entry_matches:
            return []

        query_terms = _query_terms(text)
        entry_score_map = dict(entry_matches)

        base_scores = [0.90, 0.78, 0.66, 0.56, 0.48]

        results: dict[str, dict] = {}
        visited: set[str] = set()
        frontier: set[str] = set(entry_score_map)

        for hop in range(depth + 1):
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
                node_score = entry_score_map.get(node_id, hop_score) if hop == 0 else hop_score

                existing = results.get(node_id)
                if existing is None or node_score > existing["graph_score"]:
                    results[node_id] = {"section_id": node_id, "graph_score": node_score, **node_data}

                if hop < depth:
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
            score = self._neighbor_priority(neighbor, edge_attrs.get("type", ""), query_terms, edge_attrs)
            scored.append((score, neighbor))

        for neighbor in self.index.graph.predecessors(node_id):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            edge_attrs = dict(self.index.graph.edges[neighbor, node_id])
            score = self._neighbor_priority(neighbor, edge_attrs.get("type", ""), query_terms, edge_attrs)
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
        """Return likely entry nodes with normalized confidence scores.

        Uses pre-built inverted indices to avoid scanning all graph nodes per query.
        """
        text_lower = text.lower()
        query_terms = _query_terms(text)
        section_refs = _extract_section_refs(text)
        irs_prefix_refs = _extract_irs_prefix_refs(text)

        hinted_prefixes: set[str] = set()
        for term in query_terms:
            hinted_prefixes.update(KEYWORD_TO_SECTION_HINTS.get(term, set()))

        # Build candidate set from indices instead of scanning all nodes.
        candidate_nodes: set[str] = set()

        for term in query_terms:
            candidate_nodes.update(self._term_to_nodes.get(term, set()))

        section_ref_candidates: set[str] = set()
        for ref in section_refs:
            matches = self._section_ref_to_nodes.get(ref, set())
            section_ref_candidates.update(matches)
            candidate_nodes.update(matches)

        for prefix in hinted_prefixes:
            candidate_nodes.update(self._section_ref_to_nodes.get(prefix, set()))

        for prefix in irs_prefix_refs:
            for node_id_lower, node_id in self._irs_nodes:
                if node_id_lower.startswith(prefix):
                    candidate_nodes.add(node_id)

        # Fallback: short query with no terms, try title substring match.
        if not candidate_nodes and not query_terms and text_lower:
            for node_id, (_, title, _) in self._node_text.items():
                if text_lower in title:
                    candidate_nodes.add(node_id)

        if not candidate_nodes:
            return []

        # Score candidate nodes.
        scored_matches: dict[str, float] = {}
        for node_id in candidate_nodes:
            node_id_lower, title, source_lower = self._node_text[node_id]
            score = 0.0
            overlap = sum(1 for term in query_terms if term in node_id_lower or term in title)

            if node_id in section_ref_candidates:
                score += 6.0

            if irs_prefix_refs and any(node_id_lower.startswith(p) for p in irs_prefix_refs):
                score += 2.5
                score += min(2.0, 0.5 * overlap)

            if hinted_prefixes and any(
                node_id_lower.startswith(p) for p in hinted_prefixes
            ):
                score += 4.0

            if overlap > 0:
                score += min(3.0, float(overlap))

            if score == 0.0 and not query_terms and text_lower in title:
                score += 0.5

            if score > 0.0:
                if source_lower == "usc26":
                    score += 0.1
                scored_matches[node_id] = score

        if not scored_matches:
            return []

        # When explicit section refs are present, restrict to ref-matching nodes only.
        if section_refs and section_ref_candidates:
            scored_matches = {
                nid: s for nid, s in scored_matches.items() if nid in section_ref_candidates
            }
            if not scored_matches:
                return []

        max_raw = max(scored_matches.values())
        threshold = max(cfg.GRAPH_ENTRY_THRESHOLD_BASE, max_raw - 2.0)
        if section_refs:
            threshold = max(threshold, cfg.GRAPH_ENTRY_THRESHOLD_SECTION)

        filtered = [(nid, s) for nid, s in scored_matches.items() if s >= threshold]
        if not filtered:
            # Nothing cleared threshold; take the best few rather than everything.
            filtered = sorted(scored_matches.items(), key=lambda x: (-x[1], x[0]))[:5]

        ranked = sorted(filtered, key=lambda x: (-x[1], x[0]))[:MAX_ENTRY_NODES]

        max_raw = max(s for _, s in ranked)
        return [(nid, min(1.0, 0.60 + 0.40 * (s / max_raw))) for nid, s in ranked]

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
    return [
        word
        for word in re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
        if word not in STOPWORDS
    ]


def _extract_section_refs(text: str) -> set[str]:
    """Extract canonical lowercase 26 USC section references from query text."""
    refs: set[str] = set()

    for start_raw, end_raw in _SECTION_RANGE_RE.findall(text):
        start, end = int(start_raw), int(end_raw)
        if start <= end and (end - start) <= 60:
            for value in range(start, end + 1):
                refs.add(f"26 usc §{value}")

    for num, sub in _SECTION_REF_RE.findall(text):
        num_norm = num.lower()
        if sub:
            refs.add(f"26 usc §{num_norm}{sub.lower()}")
        refs.add(f"26 usc §{num_norm}")

    return refs


def _section_ref_matches_node(section_ref: str, node_id_lower: str) -> bool:
    """Return True when a query section ref matches a graph node section id."""
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
