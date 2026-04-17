"""Analyze saved evaluation JSON files without rerunning models.

Usage examples:
  python evaluation/analyze_saved_results.py
  python evaluation/analyze_saved_results.py --dataset sara_v3 --model ollama
  python evaluation/analyze_saved_results.py --results-dir evaluation/results --show-case-winners
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_FILENAME_RE = re.compile(r"^(?P<dataset>.+)__(?P<model>.+)__(?P<mode>[^_]+)\.json$")


def _avg(values: list[float | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _safe_num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cases = payload.get("cases", [])
    scoring = [case.get("scoring", {}) for case in cases]
    retrieval = [case.get("retrieval_metrics", {}) for case in cases]

    earned = [_safe_num(item.get("earned")) for item in scoring]
    total = [_safe_num(item.get("total")) for item in scoring]
    earned_sum = sum(value for value in earned if value is not None)
    total_sum = sum(value for value in total if value is not None)

    labels = [
        str(item.get("predicted_label", "")).lower()
        for item in scoring
        if item.get("predicted_label") is not None
    ]
    unknown_label_rate = (labels.count("unknown") / len(labels)) if labels else None

    return {
        "n_cases": len(cases),
        "score_pct": (earned_sum / total_sum * 100.0) if total_sum > 0 else None,
        "score_before_citation_penalty": _avg([
            _safe_num(item.get("score_before_citation_penalty")) for item in scoring
        ]),
        "score_after_citation_penalty": _avg([
            _safe_num(item.get("score_after_citation_penalty")) for item in scoring
        ]),
        "citation_penalty": _avg([
            _safe_num(item.get("citation_penalty")) for item in scoring
        ]),
        "answer_correct": _avg([_safe_num(item.get("answer_correct")) for item in scoring]),
        "citation_fact_precision": _avg([
            _safe_num(item.get("citation_fact_precision")) for item in scoring
        ]),
        "citation_fact_recall": _avg([
            _safe_num(item.get("citation_fact_recall")) for item in scoring
        ]),
        "unknown_label_rate": unknown_label_rate,
        "recall_at_k_hier": _avg([_safe_num(item.get("recall_at_k")) for item in retrieval]),
        "mrr_hier": _avg([_safe_num(item.get("mrr")) for item in retrieval]),
        "recall_at_k_exact": _avg([
            _safe_num(item.get("recall_at_k_exact")) for item in retrieval
        ]),
        "mrr_exact": _avg([_safe_num(item.get("mrr_exact")) for item in retrieval]),
    }


def _load_results(results_dir: Path) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}

    for path in sorted(results_dir.glob("*.json")):
        match = _FILENAME_RE.match(path.name)
        if not match:
            continue

        dataset = match.group("dataset")
        model = match.group("model")
        mode = match.group("mode")

        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault((dataset, model), {})[mode] = payload

    return grouped


def _print_group_report(
    dataset: str,
    model: str,
    mode_payloads: dict[str, dict[str, Any]],
    show_case_winners: bool,
) -> None:
    print(f"\n=== {dataset} | {model} ===")

    mode_order = ["none", "vector", "graph", "hybrid"]
    present_modes = [mode for mode in mode_order if mode in mode_payloads]
    summaries = {mode: _summarize_payload(mode_payloads[mode]) for mode in present_modes}

    print(
        "mode    cases  score%   pre_pen post_pen cit_pen ans_ok  cite_prec cite_rec  "
        "unk_rate  r@k_hier mrr_hier r@k_exact mrr_exact"
    )
    for mode in present_modes:
        summary = summaries[mode]
        print(
            f"{mode:<7} {summary['n_cases']:<5} "
            f"{_fmt(summary['score_pct']):>7} "
            f"{_fmt(summary['score_before_citation_penalty']):>8} "
            f"{_fmt(summary['score_after_citation_penalty']):>8} "
            f"{_fmt(summary['citation_penalty']):>7} "
            f"{_fmt(summary['answer_correct']):>7} "
            f"{_fmt(summary['citation_fact_precision']):>9} "
            f"{_fmt(summary['citation_fact_recall']):>8} "
            f"{_fmt(summary['unknown_label_rate']):>9} "
            f"{_fmt(summary['recall_at_k_hier']):>9} "
            f"{_fmt(summary['mrr_hier']):>8} "
            f"{_fmt(summary['recall_at_k_exact']):>10} "
            f"{_fmt(summary['mrr_exact']):>9}"
        )

    if "hybrid" in summaries and "none" in summaries:
        print("\nHybrid - None Delta")
        keys = [
            "score_pct",
            "score_before_citation_penalty",
            "score_after_citation_penalty",
            "citation_penalty",
            "answer_correct",
            "citation_fact_precision",
            "citation_fact_recall",
            "unknown_label_rate",
            "recall_at_k_hier",
            "mrr_hier",
            "recall_at_k_exact",
            "mrr_exact",
        ]
        for key in keys:
            hybrid_value = summaries["hybrid"].get(key)
            none_value = summaries["none"].get(key)
            if hybrid_value is None or none_value is None:
                print(f"  {key}: -")
                continue
            delta = hybrid_value - none_value
            sign = "+" if delta >= 0 else ""
            print(f"  {key}: {sign}{delta:.4f}")

    if not show_case_winners:
        return

    if not all(mode in mode_payloads for mode in mode_order):
        return

    print("\nCase-level winner by earned score")
    case_ids = [case.get("id") for case in mode_payloads["none"].get("cases", [])]
    for case_id in case_ids:
        scores: dict[str, float] = {}
        for mode in mode_order:
            case = next(
                (item for item in mode_payloads[mode].get("cases", []) if item.get("id") == case_id),
                None,
            )
            if case is None:
                continue
            earned = _safe_num(case.get("scoring", {}).get("earned"))
            scores[mode] = earned if earned is not None else float("-inf")

        if not scores:
            continue

        winner = max(scores, key=scores.get)
        pretty = ", ".join(
            f"{mode}:{value if value != float('-inf') else 'NA'}"
            for mode, value in scores.items()
        )
        print(f"  {case_id}: winner={winner} ({pretty})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze saved evaluation JSON result files.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory containing result JSON files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional dataset filter (for example: sara_v3).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional model/tag filter (for example: ollama).",
    )
    parser.add_argument(
        "--show-case-winners",
        action="store_true",
        help="Also print per-case mode winner by earned score.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = _load_results(args.results_dir)

    if not grouped:
        raise SystemExit(f"No result JSON files found in {args.results_dir}")

    dataset_filter = args.dataset.strip().lower()
    model_filter = args.model.strip().lower()

    selected = []
    for (dataset, model), payloads in grouped.items():
        if dataset_filter and dataset.lower() != dataset_filter:
            continue
        if model_filter and model_filter not in model.lower():
            continue
        selected.append((dataset, model, payloads))

    if not selected:
        raise SystemExit("No result groups matched the selected filters.")

    for dataset, model, payloads in sorted(selected):
        _print_group_report(dataset, model, payloads, args.show_case_winners)


if __name__ == "__main__":
    main()
