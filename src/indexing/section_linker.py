"""Section linking for the tax knowledge graph.

This module has two complementary strategies:

1. Curated links (high-precision priors)
   - Hand-maintained section coverage and cross-publication pairs.

2. Inferred links (full-corpus coverage)
   - Derived from observed xref edges after normalization so every source in
     the knowledge folder can participate without manual table updates.

All injected edges use type="coverage" so provenance remains separate from
structural hierarchy edges and direct xref edges.
"""

from collections import Counter, defaultdict
import re

import networkx as nx

from src import config as cfg


# ---------------------------------------------------------------------------
# Curated priors: high-value IRS source <-> IRC section relationships
# ---------------------------------------------------------------------------

SECTION_COVERAGE: dict[str, list[str]] = {
    "26 USC §1": ["p17", "i1040gi"],
    "26 USC §21": ["i2441", "p503"],
    "26 USC §22": ["i1040gi"],
    "26 USC §24": ["i1040s8"],
    "26 USC §25A": ["i8863", "p970"],
    "26 USC §32": ["p596"],
    "26 USC §61": ["p525", "p17"],
    "26 USC §62": ["p17", "i1040gi"],
    "26 USC §63": ["p501", "p17"],
    "26 USC §68": ["p17", "i1040gi"],
    "26 USC §72": ["i8606"],
    "26 USC §86": ["p17"],
    "26 USC §101": ["p525"],
    "26 USC §102": ["p525"],
    "26 USC §104": ["p525"],
    "26 USC §105": ["p525"],
    "26 USC §108": ["p525"],
    "26 USC §85": ["i1099g", "p525"],
    "26 USC §223": ["i1099sa"],
    "26 USC §121": ["p523"],
    "26 USC §151": ["p501"],
    "26 USC §152": ["p501"],
    "26 USC §162": ["i1040sc"],
    "26 USC §163": ["p530", "p936"],
    "26 USC §164": ["p17"],
    "26 USC §165": ["p544"],
    "26 USC §170": ["p526"],
    "26 USC §213": ["p502"],
    "26 USC §219": ["i8606"],
    "26 USC §401": ["p560"],
    "26 USC §408": ["i8606"],
    "26 USC §529": ["i1099q", "i1099qa", "p970"],
    "26 USC §529A": ["i1099qa"],
    "26 USC §469": ["i1040se"],
    "26 USC §1001": ["p544", "i8949"],
    "26 USC §1211": ["i1040sd"],
    "26 USC §1212": ["i1040sd"],
    "26 USC §1221": ["p544", "i1040sd"],
    "26 USC §1222": ["i1040sd"],
    "26 USC §1223": ["i1040sd"],
}


IRS_CROSS_PUBLICATION: list[tuple[str, str]] = [
    ("p17", "i1040gi"),
    ("p17", "p596"),
    ("p17", "p501"),
    ("p17", "p502"),
    ("p17", "p503"),
    ("p17", "p526"),
    ("p17", "p523"),
    ("p17", "p970"),
    ("p17", "p529"),
    ("p17", "p505"),
    ("p17", "p936"),
    ("i1040gi", "i1040sc"),
    ("i1040gi", "i1040sd"),
    ("i1040gi", "i1040se"),
    ("i1040gi", "i1040sb"),
    ("i1040gi", "p596"),
    ("i1040gi", "i8863"),
    ("i1040gi", "i2441"),
    ("i1040gi", "i1040s8"),
    ("i1040gi", "i8995"),
    ("p596", "i1040gi"),
    ("p596", "i1040sse"),
    ("p970", "i8863"),
    ("i1040sc", "i8829"),
    ("i1040sc", "i4562"),
    ("i1040sd", "i8949"),
    ("i8606", "p560"),
    ("i1099g", "p525"),
    ("i1099qa", "i1099q"),
    ("i1099sa", "i1099q"),
]


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

_MAX_NODES_PER_SOURCE = cfg.LINKER_MAX_NODES_PER_SOURCE
_MAX_XPUB_REP_NODES = cfg.LINKER_MAX_XPUB_REP_NODES
_MAX_TARGET_SECTIONS_PER_SOURCE = cfg.LINKER_MAX_TARGET_SECTIONS
_MIN_SECTION_REF_SUPPORT = cfg.LINKER_MIN_SECTION_REF_SUPPORT
_MIN_CROSS_PUB_SUPPORT = cfg.LINKER_MIN_CROSS_PUB_SUPPORT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inject_coverage_edges(graph: nx.DiGraph) -> int:
    """Inject curated IRS-source <-> IRC-section coverage edges.

    Returns number of directed edges added.
    """
    nodes_by_source = _build_nodes_by_source(graph)
    added = 0

    for irc_prefix, irs_sources in SECTION_COVERAGE.items():
        irc_root = _find_irc_root(graph, irc_prefix)
        if irc_root is None:
            continue

        irc_topic_words = _topic_words(
            irc_prefix + " " + graph.nodes[irc_root].get("title", "")
        )

        for source_key in irs_sources:
            candidates = nodes_by_source.get(source_key, [])
            if not candidates:
                continue

            ranked = sorted(
                candidates,
                key=lambda node_id: _overlap(
                    irc_topic_words,
                    _topic_words(
                        graph.nodes[node_id].get("title", "")
                        + " "
                        + graph.nodes[node_id].get("snippet", "")
                    ),
                ),
                reverse=True,
            )

            for irs_node in ranked[:_MAX_NODES_PER_SOURCE]:
                added += _add_coverage_edge(graph, irc_root, irs_node, bidirectional=True)

    return added


def inject_cross_publication_edges(graph: nx.DiGraph) -> int:
    """Inject curated directional IRS-source -> IRS-source coverage edges.

    Returns number of directed edges added.
    """
    nodes_by_source = _build_nodes_by_source(graph)
    added = 0

    for source_a, source_b in IRS_CROSS_PUBLICATION:
        nodes_a = nodes_by_source.get(source_a, [])
        nodes_b = nodes_by_source.get(source_b, [])
        if not nodes_a or not nodes_b:
            continue

        reps_a = _representative_nodes(graph, nodes_a, _MAX_NODES_PER_SOURCE)
        reps_b = _representative_nodes(graph, nodes_b, _MAX_NODES_PER_SOURCE)

        for node_a in reps_a:
            for node_b in reps_b:
                if node_a == node_b:
                    continue
                added += _add_coverage_edge(graph, node_a, node_b, bidirectional=False)

    return added


def inject_inferred_section_coverage_edges(graph: nx.DiGraph) -> int:
    """Infer IRS-source <-> IRC-section coverage edges from observed xref links.

    This provides broad coverage for all sources in the knowledge folder,
    not only those listed in curated tables.

    Returns number of directed edges added.
    """
    nodes_by_source = _build_nodes_by_source(graph)
    counts_by_source: dict[str, Counter[str]] = defaultdict(Counter)

    for src_node, dst_node, edge_data in graph.edges(data=True):
        if edge_data.get("type") != "xref":
            continue

        src_source = graph.nodes[src_node].get("source", "")
        dst_source = graph.nodes[dst_node].get("source", "")

        if not src_source or src_source == "usc26":
            continue
        if dst_source != "usc26":
            continue

        usc_root = _usc_root_prefix(dst_node)
        if usc_root:
            counts_by_source[src_source][usc_root] += 1

    added = 0
    for source_key, section_counter in counts_by_source.items():
        source_nodes = nodes_by_source.get(source_key, [])
        if not source_nodes:
            continue

        ranked_sections = sorted(
            section_counter.items(),
            key=lambda pair: (-pair[1], pair[0]),
        )
        strong_sections = [
            (section_id, count)
            for section_id, count in ranked_sections
            if count >= _MIN_SECTION_REF_SUPPORT
        ]

        if strong_sections:
            selected_sections = strong_sections[:_MAX_TARGET_SECTIONS_PER_SOURCE]
        else:
            selected_sections = ranked_sections[: min(2, _MAX_TARGET_SECTIONS_PER_SOURCE)]

        reps = _representative_nodes(graph, source_nodes, _MAX_NODES_PER_SOURCE)

        for section_prefix, _support in selected_sections:
            irc_root = _find_irc_root(graph, section_prefix)
            if irc_root is None:
                continue
            for rep_node in reps:
                added += _add_coverage_edge(graph, irc_root, rep_node, bidirectional=True)

    return added


def inject_inferred_cross_publication_edges(graph: nx.DiGraph) -> int:
    """Infer directional IRS-source -> IRS-source links from xref evidence.

    This scales cross-publication linking to all forms/publications/schedules
    represented in the graph.

    Returns number of directed edges added.
    """
    nodes_by_source = _build_nodes_by_source(graph)
    pair_counter: Counter[tuple[str, str]] = Counter()

    for src_node, dst_node, edge_data in graph.edges(data=True):
        if edge_data.get("type") != "xref":
            continue

        src_source = graph.nodes[src_node].get("source", "")
        dst_source = graph.nodes[dst_node].get("source", "")

        if not src_source or not dst_source:
            continue
        if src_source == "usc26" or dst_source == "usc26":
            continue
        if src_source == dst_source:
            continue

        pair_counter[(src_source, dst_source)] += 1

    added = 0
    for (source_a, source_b), support in pair_counter.most_common():
        if support < _MIN_CROSS_PUB_SUPPORT:
            continue

        nodes_a = nodes_by_source.get(source_a, [])
        nodes_b = nodes_by_source.get(source_b, [])
        if not nodes_a or not nodes_b:
            continue

        reps_a = _representative_nodes(graph, nodes_a, _MAX_XPUB_REP_NODES)
        reps_b = _representative_nodes(graph, nodes_b, _MAX_XPUB_REP_NODES)

        for node_a in reps_a:
            for node_b in reps_b:
                if node_a == node_b:
                    continue
                added += _add_coverage_edge(graph, node_a, node_b, bidirectional=False)

    return added


def inject_fallback_connectivity_edges(
    graph: nx.DiGraph,
    min_xref_coverage_edges: int,
) -> int:
    """Backstop links for sources that remain isolated after normal linking.

    For any non-USC source with fewer than ``min_xref_coverage_edges`` xref/coverage
    edges, connect representative source nodes to the best-matching USC root.
    This keeps graph audit robust for sparse forms and short PDFs.
    """
    nodes_by_source = _build_nodes_by_source(graph)
    usc_roots = _usc_root_nodes(graph)
    if not usc_roots:
        return 0

    added = 0
    for source_key, source_nodes in nodes_by_source.items():
        if source_key in {"", "usc26"}:
            continue
        if not source_nodes:
            continue

        edge_count = _source_xref_coverage_edge_count(graph, source_nodes)
        if edge_count >= min_xref_coverage_edges:
            continue

        reps = _representative_nodes(graph, source_nodes, max(1, _MAX_NODES_PER_SOURCE))
        root = _best_matching_usc_root(graph, reps, usc_roots)
        if root is None:
            root = _fallback_usc_root(graph, usc_roots)
        if root is None:
            continue

        for rep_node in reps[:2]:
            added += _add_coverage_edge(graph, root, rep_node, bidirectional=True)

    return added


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_nodes_by_source(graph: nx.DiGraph) -> dict[str, list[str]]:
    nodes_by_source: dict[str, list[str]] = {}
    for node_id in graph.nodes:
        source = graph.nodes[node_id].get("source", "")
        nodes_by_source.setdefault(source, []).append(node_id)
    return nodes_by_source


def _source_xref_coverage_edge_count(graph: nx.DiGraph, node_ids: list[str]) -> int:
    count = 0
    for node_id in node_ids:
        for neighbor in graph.successors(node_id):
            edge_type = graph.edges[node_id, neighbor].get("type", "")
            if edge_type in {"xref", "coverage"}:
                count += 1
        for neighbor in graph.predecessors(node_id):
            edge_type = graph.edges[neighbor, node_id].get("type", "")
            if edge_type in {"xref", "coverage"}:
                count += 1
    return count


def _usc_root_nodes(graph: nx.DiGraph) -> list[str]:
    roots = [
        node_id
        for node_id in graph.nodes
        if graph.nodes[node_id].get("source") == "usc26"
        and re.match(r"^26 USC §\d+[A-Z]?$", node_id)
    ]
    return sorted(roots, key=len)


def _best_matching_usc_root(
    graph: nx.DiGraph,
    source_nodes: list[str],
    usc_roots: list[str],
) -> str | None:
    source_text = []
    for node_id in source_nodes:
        source_text.append(graph.nodes[node_id].get("title", ""))
        source_text.append(graph.nodes[node_id].get("snippet", ""))
    source_words = _topic_words(" ".join(source_text))

    best_root = None
    best_score = -1
    for root in usc_roots:
        root_words = _topic_words(root + " " + graph.nodes[root].get("title", ""))
        score = _overlap(source_words, root_words)
        if score > best_score:
            best_score = score
            best_root = root

    if best_score <= 0:
        return None
    return best_root


def _fallback_usc_root(graph: nx.DiGraph, usc_roots: list[str]) -> str | None:
    for preferred in ("26 USC §61", "26 USC §63", "26 USC §1"):
        if preferred in graph:
            return preferred
    return usc_roots[0] if usc_roots else None


def _representative_nodes(graph: nx.DiGraph, node_ids: list[str], k: int) -> list[str]:
    """Pick representative source nodes (high-level/intro-like nodes)."""
    return sorted(
        node_ids,
        key=lambda node_id: (
            len(graph.nodes[node_id].get("title", "")),
            len(node_id),
            node_id,
        ),
    )[:k]


def _add_coverage_edge(
    graph: nx.DiGraph,
    source_node: str,
    target_node: str,
    bidirectional: bool,
) -> int:
    """Add one or two coverage edges if they do not already exist."""
    added = 0

    if source_node != target_node and not graph.has_edge(source_node, target_node):
        graph.add_edge(source_node, target_node, type="coverage")
        added += 1

    if bidirectional and source_node != target_node and not graph.has_edge(target_node, source_node):
        graph.add_edge(target_node, source_node, type="coverage")
        added += 1

    return added


def _usc_root_prefix(section_id: str) -> str | None:
    """Return root USC section prefix, e.g. '26 USC §32' from '26 USC §32(c)(2)'."""
    match = re.match(r"^(26 USC §\d+[A-Z]?)", section_id)
    if not match:
        return None
    return match.group(1)


def _find_irc_root(graph: nx.DiGraph, section_prefix: str) -> str | None:
    """Return a USC node matching the given section prefix."""
    if section_prefix in graph:
        return section_prefix

    boundary_re = re.compile(rf"^{re.escape(section_prefix)}(?:\b|\(|$)")
    matches = [
        node_id
        for node_id in graph.nodes
        if boundary_re.match(node_id) and graph.nodes[node_id].get("source") == "usc26"
    ]
    if not matches:
        return None
    return min(matches, key=len)


def _topic_words(text: str) -> frozenset[str]:
    return frozenset(word.lower() for word in re.findall(r"[a-zA-Z]{4,}", text))


def _overlap(words_a: frozenset[str], words_b: frozenset[str]) -> int:
    return len(words_a & words_b)
