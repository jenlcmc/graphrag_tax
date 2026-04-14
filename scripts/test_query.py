"""Smoke-test the retrieval pipeline with a single query.

Run after build_pipeline.py has created the data/processed/ artifacts.

Usage:
  python scripts/test_query.py
  python scripts/test_query.py "What is the standard deduction for single filers?"
  python scripts/test_query.py "401k contribution limit" --mode vector
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config as cfg
from src.retrieval.hybrid_retriever import HybridRetriever

DEFAULT_QUERY = "What is the standard deduction for a single filer in 2024?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--mode", choices=["none", "vector", "graph", "hybrid"], default="hybrid")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    print(f"Query : {args.query}")
    print(f"Mode  : {args.mode}")
    print(f"Top-k : {args.k}")
    print("-" * 60)

    retriever = HybridRetriever.load(cfg)
    results   = retriever.query(args.query, k=args.k, mode=args.mode)

    if not results:
        print("No results returned.")
        return

    for i, r in enumerate(results, 1):
        score = r.get("score", r.get("vector_score", r.get("graph_score", 0.0)))
        print(f"\n[{i}] {r['section_id']}  (score={score:.3f})")
        print(f"     Source : {r.get('source', '')}")
        print(f"     Title  : {r.get('title', '')}")
        text = r.get("text", r.get("snippet", ""))
        print(f"     Text   : {text[:200]}...")


if __name__ == "__main__":
    main()
