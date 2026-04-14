import networkx as nx

from src.indexing.section_linker import inject_fallback_connectivity_edges


def test_fallback_connectivity_adds_edges_for_isolated_source():
    graph = nx.DiGraph()
    graph.add_node("26 USC §61", source="usc26", title="Gross income", snippet="gross income")
    graph.add_node("IRS Pub. 999: Intro", source="p999", title="Intro", snippet="income example")

    added = inject_fallback_connectivity_edges(graph, min_xref_coverage_edges=1)

    assert added >= 2
    assert graph.has_edge("26 USC §61", "IRS Pub. 999: Intro")
    assert graph.has_edge("IRS Pub. 999: Intro", "26 USC §61")
    assert graph.edges["26 USC §61", "IRS Pub. 999: Intro"]["type"] == "coverage"


def test_fallback_connectivity_skips_already_connected_source():
    graph = nx.DiGraph()
    graph.add_node("26 USC §61", source="usc26", title="Gross income", snippet="gross income")
    graph.add_node("IRS Pub. 999: Intro", source="p999", title="Intro", snippet="income example")
    graph.add_edge("IRS Pub. 999: Intro", "26 USC §61", type="xref")

    added = inject_fallback_connectivity_edges(graph, min_xref_coverage_edges=1)

    assert added == 0
