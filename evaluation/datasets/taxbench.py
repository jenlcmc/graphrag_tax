"""TaxBench dataset adapter.

Loads TaxBench-EvalSet.jsonl (114 tax law Q&A cases).  Each case has a
question and a rubric string describing the ideal answer with per-criterion
point values, e.g.:

  [+0.10] Answer explains the Green Card test correctly.
  [+0.20] Identifies the substantial presence test threshold.

Two complementary scores are computed for each response:

  LLM-as-judge  — semantic; the judge assigns partial credit per criterion.
  ROUGE (1/2/L) — lexical; overlap between the response and the rubric text
                  (with point annotations stripped).  Provides a fast,
                  reproducible signal that requires no API call.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evaluation.datasets.base import Dataset, EvalCase
from src.utils.ref_patterns import USC_SECTION_RE, USC_SECTION_RANGE_RE


# ---------------------------------------------------------------------------
# Rubric helpers
# ---------------------------------------------------------------------------

# Matches point annotations like [+0.10] or [+1.00].
_CRITERION_RE = re.compile(r"\[\+(\d+\.\d+)\]")


def _rubric_max_score(rubric: str) -> float:
    """Return the total points available in the rubric."""
    return sum(float(m) for m in _CRITERION_RE.findall(rubric))


def _rubric_as_reference(rubric: str) -> str:
    """Strip [+N.NN] annotations to produce a plain reference text for ROUGE."""
    return _CRITERION_RE.sub("", rubric).strip()


def _extract_rubric_section_refs(rubric: str) -> list[str]:
    """Extract USC section IDs mentioned in the rubric as pseudo ground-truth.

    Returns lowercase IDs matching the graph node ID format, e.g. "26 usc §32".
    Used to compute recall@k and MRR against retrieved section IDs.
    """
    refs: set[str] = set()

    for start_raw, end_raw in USC_SECTION_RANGE_RE.findall(rubric):
        start, end = int(start_raw), int(end_raw)
        if start <= end and (end - start) <= 60:
            for val in range(start, end + 1):
                refs.add(f"26 usc §{val}")

    for num, *_ in USC_SECTION_RE.findall(rubric):
        refs.add(f"26 usc §{num}")

    return sorted(refs)


# ---------------------------------------------------------------------------
# ROUGE scoring
# ---------------------------------------------------------------------------

def _compute_rouge(prediction: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Returns a dict with keys rouge1, rouge2, rougeL (all floats in [0, 1]).
    Falls back to zeros if rouge_score is not installed.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------

class TaxBenchDataset(Dataset):
    """Adapter for TaxBench-EvalSet.jsonl."""

    name = "taxbench"

    def load(self, data_dir: Path, limit: int | None = None) -> list[EvalCase]:
        path = data_dir / "TaxBench-EvalSet.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"TaxBench dataset not found at {path}. "
                "Expected TaxBench-EvalSet.jsonl inside the dataset/ directory."
            )

        cases: list[EvalCase] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rubric = row.get("answer_rubric", "")
                cases.append(
                    EvalCase(
                        id=row["id"],
                        question=row["question"],
                        rubric=rubric,
                        relevant_ids=_extract_rubric_section_refs(rubric),
                        metadata={
                            "max_score": _rubric_max_score(rubric),
                        },
                    )
                )
                if limit is not None and len(cases) >= limit:
                    break

        return cases

    def score(self, response: str, case: EvalCase, judge_fn) -> dict:
        """Score a response using both LLM-as-judge and ROUGE.

        Returns a dict merging both signal types:
          earned, total, feedback  — from LLM-as-judge (may be None if skipped)
          rouge1, rouge2, rougeL   — F1 scores vs. rubric reference text
        """
        # LLM-as-judge (calls super which invokes judge_fn)
        judge_result = super().score(response, case, judge_fn)

        # ROUGE vs. rubric reference text (no API call needed)
        rouge_result: dict = {}
        if case.rubric:
            reference = _rubric_as_reference(case.rubric)
            rouge_result = _compute_rouge(response, reference)

        return {**judge_result, **rouge_result}
