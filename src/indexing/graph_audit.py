"""Graph quality and coverage checks for build-time validation."""

from collections import defaultdict
import json
from pathlib import Path

import networkx as nx


def audit_graph_coverage(
    graph: nx.DiGraph,
    min_xref_coverage_edges: int = 1,
) -> dict:
    """Audit source connectivity and reference-edge coverage.

    Checks:
    1. Source-level connectivity: every source must connect to at least one
       node from another source.
    2. Source-level reference coverage: every source must have at least
       ``min_xref_coverage_edges`` edges of type xref or coverage.

    Returns a serializable report.
    """
    undirected = graph.to_undirected()

    component_map: dict[str, int] = {}
    component_members: dict[int, set[str]] = {}
    for component_id, nodes in enumerate(nx.connected_components(undirected)):
        members = set(nodes)
        component_members[component_id] = members
        for node_id in members:
            component_map[node_id] = component_id

    nodes_by_source: dict[str, list[str]] = defaultdict(list)
    for node_id in graph.nodes:
        source = str(graph.nodes[node_id].get("source", ""))
        if source:
            nodes_by_source[source].append(node_id)

    per_source: dict[str, dict] = {}
    issues: list[str] = []

    for source in sorted(nodes_by_source):
        source_nodes = nodes_by_source[source]
        xref_coverage_edge_count = 0
        external_sources: set[str] = set()

        for node_id in source_nodes:
            for neighbor in graph.successors(node_id):
                edge_type = graph.edges[node_id, neighbor].get("type", "")
                if edge_type in {"xref", "coverage"}:
                    xref_coverage_edge_count += 1
                neighbor_source = str(graph.nodes[neighbor].get("source", ""))
                if neighbor_source and neighbor_source != source:
                    external_sources.add(neighbor_source)

            for neighbor in graph.predecessors(node_id):
                edge_type = graph.edges[neighbor, node_id].get("type", "")
                if edge_type in {"xref", "coverage"}:
                    xref_coverage_edge_count += 1
                neighbor_source = str(graph.nodes[neighbor].get("source", ""))
                if neighbor_source and neighbor_source != source:
                    external_sources.add(neighbor_source)

        source_component_ids = {
            component_map[node_id]
            for node_id in source_nodes
            if node_id in component_map
        }

        shares_component_with_other_source = False
        for component_id in source_component_ids:
            component_sources = {
                str(graph.nodes[node_id].get("source", ""))
                for node_id in component_members[component_id]
            }
            component_sources.discard("")
            if any(other_source != source for other_source in component_sources):
                shares_component_with_other_source = True
                break

        is_disconnected = not shares_component_with_other_source or not external_sources
        has_min_reference_edges = xref_coverage_edge_count >= min_xref_coverage_edges

        per_source[source] = {
            "n_nodes": len(source_nodes),
            "n_components": len(source_component_ids),
            "external_sources_connected": sorted(external_sources),
            "xref_coverage_edges": xref_coverage_edge_count,
            "is_disconnected": is_disconnected,
            "has_min_reference_edges": has_min_reference_edges,
        }

        if is_disconnected:
            issues.append(
                f"Source '{source}' is disconnected from other sources in the graph."
            )
        if not has_min_reference_edges:
            issues.append(
                f"Source '{source}' has only {xref_coverage_edge_count} xref/coverage edges "
                f"(< {min_xref_coverage_edges})."
            )

    report = {
        "ok": len(issues) == 0,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_sources": len(per_source),
        "min_xref_coverage_edges": min_xref_coverage_edges,
        "issues": issues,
        "per_source": per_source,
    }
    return report


def save_audit_report(report: dict, output_path: Path) -> None:
    """Save audit report as JSON."""
    output_path.write_text(json.dumps(report, indent=2))
