"""Abstract base classes for evaluation datasets.

To add a new dataset:
  1. Create evaluation/datasets/<name>.py
  2. Implement Dataset (load + optionally score)
  3. Register it in the REGISTRY dict in run_eval.py

Every dataset produces a list of EvalCase objects. The runner handles
retrieval and LLM calls; the dataset controls how responses are scored.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalCase:
    """One evaluation example."""

    id: str
    question: str
    # Expected answer or scoring rubric.  None means score() will be skipped.
    rubric: str | None = None
    # Optional ground-truth section IDs (e.g. ["26 usc §32", "26 usc §152"]).
    # When populated, run_eval.py computes recall@k and MRR against retrieved IDs.
    relevant_ids: list[str] = field(default_factory=list)
    # Any dataset-specific extra data (raw input JSON, answer choices, etc.)
    metadata: dict = field(default_factory=dict)


class Dataset(ABC):
    """Base class for evaluation datasets.

    Subclasses must implement ``load``.  The default ``score`` uses
    LLM-as-judge against ``case.rubric``; override for exact-match datasets.

    ``judge_model_id`` is injected by the runner after construction so that
    adapters which call an LLM directly (e.g. with a custom judge prompt) do
    not need to hard-code a model name.
    """

    # Short name used in output filenames, e.g. "taxbench"
    name: str
    # Injected by run_eval.py; available to score() overrides that need it.
    judge_model_id: str = ""

    @abstractmethod
    def load(self, data_dir: Path, limit: int | None = None) -> list[EvalCase]:
        """Load cases from data_dir.

        Args:
            data_dir: Path to the dataset directory (e.g. research_789/dataset/).
            limit:    If set, return at most this many cases.

        Returns:
            List of EvalCase objects.
        """

    def score(self, response: str, case: EvalCase, judge_fn) -> dict:
        """Score one model response against the expected answer or rubric.

        The default implementation calls judge_fn(response, rubric) which
        uses an LLM to assign a score.  Override this for datasets where
        scoring can be done deterministically (e.g., exact-match, numeric).

        Args:
            response: The model's raw text output.
            case:     The EvalCase, including rubric.
            judge_fn: Callable(response, rubric) -> {earned, total, feedback}.

        Returns:
            Dict with at minimum:
              "earned"   - float, points earned (or None if rubric is absent)
              "total"    - float, max possible points (or None)
              "feedback" - str, brief explanation
        """
        if case.rubric is None:
            return {"earned": None, "total": None, "feedback": "No rubric provided."}

        return judge_fn(response, case.rubric)
