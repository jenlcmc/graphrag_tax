"""NetworkX knowledge graph index with GraphRAG community detection.

Graph structure (directed):
  Nodes  — one per chunk, keyed by section_id
  Edges (three types):
    "hierarchy" — parent_id → section_id, from XML tree structure
    "xref"      — section_id → cross_ref, from explicit <ref> tags and
                  §NNN / publication-name mentions extracted by the normalizer
    "coverage"  — injected by section_linker from curated priors plus inferred
                  links learned from observed xref patterns; represents the
                  "this IRS document primarily explains/relates to this target"
                  relationship

The three edge types together enable multi-hop BFS chains that flat vector
retrieval cannot traverse.  For example:

  User asks about EIC
  ──────────────────
  Entry nodes: "IRS Pub. 596: *" (keyword match) + "26 USC §32" (§-ref match)
  BFS hop 1:   "26 USC §32(a)", "26 USC §32(b)", "26 USC §32(c)"  ← hierarchy
               "IRS Pub. 596: Introduction"                        ← coverage
  BFS hop 2:   "IRS Pub. 596: Who Can Claim the EIC"              ← hierarchy
               "26 USC §32" from Pub 596 chunk                     ← xref

Community detection (following Edge et al. 2024):
  Louvain on the undirected projection groups related provisions.  Each
  community gets an extractive summary for broad-query answering.

Files written:
  graph.graphml        — full graph (nodes + edges)
  communities.json     — {community_id: {members, summary, keywords}}
"""

import json
import re
import networkx as nx
from pathlib import Path

from src import config as cfg
from src.indexing.section_linker import (
    inject_coverage_edges,
    inject_cross_publication_edges,
    inject_fallback_connectivity_edges,
    inject_inferred_cross_publication_edges,
    inject_inferred_section_coverage_edges,
)


class GraphIndex:
    def __init__(self):
        self.graph       = nx.DiGraph()
        self.communities: dict[int, dict] = {}

    def build(self, chunks: list[dict]) -> None:
        """Build graph from chunks: add nodes, edges, coverage links, communities."""
        self._add_nodes(chunks)
        known_ids    = set(self.graph.nodes)
        prefix_index = _build_prefix_index(self.graph)
        self._add_edges(chunks, known_ids, prefix_index)

        n_cov_curated = inject_coverage_edges(self.graph)
        n_xpub_curated = inject_cross_publication_edges(self.graph)
        n_cov_inferred = inject_inferred_section_coverage_edges(self.graph)
        n_xpub_inferred = inject_inferred_cross_publication_edges(self.graph)
        n_fallback = 0
        if cfg.LINKER_ENABLE_FALLBACK_CONNECTIVITY:
            n_fallback = inject_fallback_connectivity_edges(
                self.graph,
                min_xref_coverage_edges=cfg.LINKER_FALLBACK_MIN_XREF_COVERAGE,
            )
        print(
            "  Section linker added "
            f"{n_cov_curated} curated-coverage + "
            f"{n_xpub_curated} curated-cross-pub + "
            f"{n_cov_inferred} inferred-coverage + "
            f"{n_xpub_inferred} inferred-cross-pub + "
            f"{n_fallback} fallback edges"
        )

        self._detect_communities()

    def save(self, graph_path: Path, community_path: Path) -> None:
        nx.write_graphml(self.graph, str(graph_path))

        serialisable = {str(cid): data for cid, data in self.communities.items()}
        community_path.write_text(json.dumps(serialisable, indent=2))

        print(
            f"Saved graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges -> {graph_path}"
        )
        print(f"Saved {len(self.communities)} communities -> {community_path}")

    def load(self, graph_path: Path, community_path: Path) -> None:
        self.graph       = nx.read_graphml(str(graph_path))
        raw              = json.loads(community_path.read_text())
        self.communities = {int(k): v for k, v in raw.items()}

    def get_neighbors(self, section_id: str, depth: int = 2) -> list[dict]:
        """BFS from section_id; return all nodes reachable within `depth` hops.

        Traverses all edge types in both directions so that coverage edges
        are followed just like structural hierarchy/xref edges.
        """
        if section_id not in self.graph:
            return []

        visited  = {section_id}
        frontier = {section_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier.update(self.graph.successors(node))
                next_frontier.update(self.graph.predecessors(node))
            frontier = next_frontier - visited
            visited.update(frontier)

        visited.discard(section_id)
        return [
            {"section_id": node, **dict(self.graph.nodes[node])}
            for node in visited
            if node in self.graph.nodes
        ]

    def get_community_for_node(self, section_id: str) -> dict | None:
        """Return the community summary containing section_id, or None."""
        if section_id not in self.graph:
            return None
        community_id = self.graph.nodes[section_id].get("community_id")
        if community_id is None:
            return None
        return self.communities.get(int(community_id))

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _add_nodes(self, chunks: list[dict]) -> None:
        for chunk in chunks:
            self.graph.add_node(
                chunk["section_id"],
                id=chunk["id"],
                title=chunk["title"],
                source=chunk["source"],
                snippet=chunk["text"][:300],
            )

    def _add_edges(
        self,
        chunks: list[dict],
        known_ids: set[str],
        prefix_index: dict[str, list[str]],
    ) -> None:
        """Add hierarchy and xref edges from chunk metadata.

        Handles two kinds of cross_refs:
          - Exact node_ids (e.g. "26 USC §32")         → direct xref edge
          - IRS label prefixes (e.g. "IRS Pub. 596")   → prefix-match xref edge
            to the shallowest (highest-level) matching node

        Prefix-based refs come from the normalizer's extract_irs_refs; exact
        refs come from extract_usc_refs and the usc26_parser's <ref> elements.
        """
        for chunk in chunks:
            sid = chunk["section_id"]

            # Hierarchy edge: parent → this chunk
            if chunk["parent_id"] and chunk["parent_id"] in known_ids:
                self.graph.add_edge(chunk["parent_id"], sid, type="hierarchy")

            for ref in chunk["cross_refs"]:
                if ref in known_ids:
                    # Exact match: standard USC or IRS section_id
                    self.graph.add_edge(sid, ref, type="xref")
                elif ref in prefix_index:
                    # Prefix match: "IRS Pub. 596" → top-level section node(s)
                    for target in prefix_index[ref][:2]:
                        self.graph.add_edge(sid, target, type="xref")

    def _detect_communities(self) -> None:
        """Louvain community detection on the undirected graph projection."""
        undirected     = self.graph.to_undirected()
        community_sets = list(nx.community.louvain_communities(undirected, seed=42))

        for community_id, members in enumerate(community_sets):
            for node_id in members:
                self.graph.nodes[node_id]["community_id"] = community_id

        for community_id, members in enumerate(community_sets):
            self.communities[community_id] = self._build_community_summary(
                community_id, members
            )

    def _build_community_summary(self, community_id: int, members: set) -> dict:
        """Extractive summary for a community.

        Titles and leading sentences from member nodes, with USC26 nodes first.
        Keywords are extracted from titles for broad-query matching.
        """
        node_data = []
        for node_id in members:
            attrs = dict(self.graph.nodes[node_id])
            attrs["section_id"] = node_id
            node_data.append(attrs)

        # Sort: IRC statute nodes first, then IRS explanations
        node_data.sort(
            key=lambda n: (0 if n.get("source") == "usc26" else 1, n["section_id"])
        )

        title_lines: list[str] = []
        text_lines:  list[str] = []
        keywords:    set[str]  = set()

        for node in node_data[:20]:
            title   = node.get("title", "")
            snippet = node.get("snippet", "")
            first   = _first_sentence(snippet)

            if title:
                title_lines.append(f"[{node['section_id']}] {title}")
            if first:
                text_lines.append(first)

            for word in re.findall(r"\b[a-zA-Z]{5,}\b", title):
                keywords.add(word.lower())

        summary = " ".join(title_lines + text_lines)
        return {
            "community_id": community_id,
            "size":         len(members),
            "members":      sorted(members),
            "summary":      summary[:1500],
            "keywords":     sorted(keywords),
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_prefix_index(graph: nx.DiGraph) -> dict[str, list[str]]:
    """Map IRS label prefixes to their top-level (shallowest) node_ids.

    IRS section_ids look like "IRS Pub. 596: What Is the EIC?".  The prefix
    is everything before the first colon: "IRS Pub. 596".  We keep the two
    shortest section_ids per prefix as representatives (shortest ≈ highest
    level in the document outline).
    """
    raw: dict[str, list[str]] = {}
    for node_id in graph.nodes:
        if ":" in node_id:
            prefix = node_id.split(":", 1)[0].strip()
            raw.setdefault(prefix, []).append(node_id)

    # Keep only the top-2 shallowest (shortest) section_ids per prefix
    return {prefix: sorted(nodes, key=len)[:2] for prefix, nodes in raw.items()}


def _first_sentence(text: str) -> str:
    """Return the first sentence of text, or the first 120 chars."""
    match = re.search(r"[.!?]", text)
    if match:
        return text[: match.start() + 1].strip()
    return text[:120].strip()
