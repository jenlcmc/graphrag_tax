"""Simple GraphML visualizer.

Reads data/processed/graph.graphml and writes an interactive HTML view.
This is a lightweight verification tool for graph structure and edge types.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize graph.graphml as HTML")
    parser.add_argument(
        "--graph-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "graph.graphml",
        help="Path to graph.graphml",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "graph_view.html",
        help="Output HTML path",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=0,
        help="Number of high-degree nodes to display (0 uses all nodes)",
    )
    return parser.parse_args()


def trim(text: str, max_len: int) -> str:
    value = (text or "").replace("\n", " ").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def build_color_map(sources: list[str]) -> dict[str, str]:
    unique_sources = sorted(set(sources))
    return {
        source: PALETTE[index % len(PALETTE)]
        for index, source in enumerate(unique_sources)
    }


def main() -> None:
    args = parse_args()
    args.graph_file = args.graph_file.expanduser().resolve()
    args.output_file = args.output_file.expanduser().resolve()

    if not args.graph_file.exists():
        raise FileNotFoundError(f"Graph file not found: {args.graph_file}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    graph = nx.read_graphml(str(args.graph_file))
    if graph.number_of_nodes() == 0:
        args.output_file.write_text("<html><body><h3>Graph is empty.</h3></body></html>")
        print(f"Wrote empty graph view -> {args.output_file}")
        return

    sample_n = graph.number_of_nodes() if args.sample_n <= 0 else args.sample_n
    sample_n = max(1, min(sample_n, graph.number_of_nodes()))
    top_nodes = [
        node_id
        for node_id, _ in sorted(graph.degree, key=lambda item: item[1], reverse=True)[:sample_n]
    ]
    subgraph = graph.subgraph(top_nodes).copy()

    sources = [str(subgraph.nodes[node_id].get("source", "unknown")) for node_id in subgraph.nodes]
    color_map = build_color_map(sources)

    nodes = []
    for node_id, attrs in subgraph.nodes(data=True):
        source = str(attrs.get("source", "unknown"))
        title = trim(str(attrs.get("title", "")), 100)
        nodes.append(
            {
                "id": str(node_id),
                "label": trim(str(node_id), 48),
                "title": f"{node_id} | {source} | {title}",
                "group": source,
                "color": color_map[source],
            }
        )

    edges = []
    for start, end, attrs in subgraph.edges(data=True):
        edges.append(
            {
                "from": str(start),
                "to": str(end),
                "label": str(attrs.get("type", "")),
                "arrows": "to",
                "font": {"size": 10},
            }
        )

    source_counts: dict[str, int] = {}
    for source in sources:
        source_counts[source] = source_counts.get(source, 0) + 1

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Graph View</title>
  <script src=\"https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js\"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 18px; }}
    #net {{ width: 100%; height: 78vh; border: 1px solid #ddd; border-radius: 8px; }}
    .meta {{ color: #444; margin-bottom: 8px; }}
    .legend-item {{ display: inline-block; margin-right: 12px; font-size: 12px; }}
    .swatch {{ width: 10px; height: 10px; display: inline-block; margin-right: 4px; border-radius: 2px; }}
  </style>
</head>
<body>
  <h2>Graph Snapshot</h2>
  <div class=\"meta\">sampled_nodes={subgraph.number_of_nodes()} total_nodes={graph.number_of_nodes()} sampled_edges={subgraph.number_of_edges()} total_edges={graph.number_of_edges()}</div>
  <div id=\"legend\"></div>
  <div id=\"net\"></div>

  <script>
    const nodes = new vis.DataSet({json.dumps(nodes)});
    const edges = new vis.DataSet({json.dumps(edges)});
    const container = document.getElementById('net');
    const data = {{ nodes, edges }};
    const options = {{
      interaction: {{ hover: true, tooltipDelay: 120 }},
      physics: {{
        stabilization: true,
        barnesHut: {{ gravitationalConstant: -3500, springLength: 110, springConstant: 0.03 }}
      }},
      edges: {{ smooth: false, color: {{ opacity: 0.35 }} }},
      nodes: {{ shape: 'dot', size: 8, borderWidth: 0 }}
    }};
    new vis.Network(container, data, options);

    const sourceCounts = {json.dumps(source_counts)};
    const sourceColors = {json.dumps(color_map)};
    const legend = document.getElementById('legend');
    for (const source of Object.keys(sourceCounts).sort()) {{
      const item = document.createElement('span');
      item.className = 'legend-item';
      item.innerHTML = `<span class=\"swatch\" style=\"background:${{sourceColors[source]}}\"></span>${{source}} (${{sourceCounts[source]}})`;
      legend.appendChild(item);
    }}
  </script>
</body>
</html>
"""

    args.output_file.write_text(html)
    print(f"Wrote graph view -> {args.output_file}")


if __name__ == "__main__":
    main()
