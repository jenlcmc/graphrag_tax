"""SARA v3 dataset adapter.

Loads Prolog-based SARA cases from dataset/sara_v3/cases and selects a split
from dataset/sara_v3/splits/{train|test}. The adapter converts each case into
an EvalCase with question text, expected answer (rubric), and extracted USC
references for retrieval metrics.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from evaluation.datasets.base import Dataset, EvalCase
from src.preprocessing.normalizer import extract_usc_refs


_LABEL_RE = re.compile(r"\b(entailment|contradiction|unknown)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
_CALC_MARKER_RE = re.compile(
    r"(?:step\s*\d+|=|\+|\-|\*|/|minus|plus|times|divide|therefore|final answer)",
    re.IGNORECASE,
)


def _clean_money(text: str) -> str:
    return re.sub(r"\s+", "", text.replace(",", "").replace("$", "")).strip()


def _extract_first_number(text: str) -> str | None:
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    return _clean_money(match.group(1))


def _extract_all_numbers(text: str) -> list[str]:
    return [_clean_money(token) for token in _NUMBER_RE.findall(text)]


def _citation_metrics_for_case(response: str, case: EvalCase) -> dict:
    """Compute citation correctness against SARA case fact/question references."""
    cited_refs = extract_usc_refs(response or "")
    cited_lower = [ref.lower() for ref in cited_refs]

    fact_refs = [ref.lower() for ref in case.relevant_ids]
    fact_ref_set = set(fact_refs)

    matched = [ref for ref in cited_refs if ref.lower() in fact_ref_set]
    matched_set = {ref.lower() for ref in matched}

    if not fact_ref_set:
        precision = 1.0 if not cited_refs else 0.0
        recall = None
    else:
        precision = (len(matched) / len(cited_refs)) if cited_refs else 0.0
        recall = len(matched_set) / len(fact_ref_set)

    return {
        "cited_refs": cited_refs,
        "matched_refs": sorted(matched_set),
        "citation_fact_precision": round(precision, 4),
        "citation_fact_recall": round(recall, 4) if recall is not None else None,
        "n_fact_refs": len(fact_ref_set),
        "n_cited_refs": len(cited_refs),
        "n_correct_cited_refs": len(matched),
    }


def _parse_case_file(path: Path) -> tuple[str, str, str, str]:
    """Parse one SARA case file.

    Returns:
      context_text, question, expected_answer, expected_type
    expected_type is one of: label, numeric, freeform.
    """
    text_lines: list[str] = []
    question_lines: list[str] = []
    section: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if line.startswith("% Text"):
            section = "text"
            continue
        if line.startswith("% Question"):
            section = "question"
            continue
        if line.startswith("% Facts") or line.startswith("% Test"):
            section = None
            continue
        if not line.startswith("%"):
            continue

        content = line[1:].strip()
        if not content:
            continue

        if section == "text":
            text_lines.append(content)
        elif section == "question":
            question_lines.append(content)

    context_text = " ".join(text_lines).strip()
    question_raw = " ".join(question_lines).strip()

    expected = ""
    expected_type = "freeform"
    question = question_raw

    if "?" in question_raw:
        before, after = question_raw.rsplit("?", 1)
        question = before.strip() + "?"
        candidate = after.strip()
        if candidate:
            expected = candidate

    if not expected:
        label_match = _LABEL_RE.search(question_raw)
        if label_match:
            expected = label_match.group(1).capitalize()
            question = question_raw[: label_match.start()].strip().rstrip(".")
            if question and not question.endswith("?"):
                question += "?"

    if not expected:
        if path.stem.endswith("_pos"):
            expected = "Entailment"
        elif path.stem.endswith("_neg"):
            expected = "Contradiction"

    if _LABEL_RE.fullmatch(expected.strip()):
        expected_type = "label"
    elif _extract_first_number(expected or "") is not None:
        expected_type = "numeric"

    if not question:
        question = question_raw

    return context_text, question, expected, expected_type


def _compute_rouge(prediction: str, reference: str) -> dict:
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


class SARAV3Dataset(Dataset):
    """Adapter for dataset/sara_v3."""

    name = "sara_v3"

    def load(self, data_dir: Path, limit: int | None = None) -> list[EvalCase]:
        root = data_dir / "sara_v3"
        cases_dir = root / "cases"
        splits_dir = root / "splits"

        split = os.getenv("SARA_SPLIT", "test").strip().lower() or "test"
        split_file = splits_dir / split

        if not cases_dir.exists():
            raise FileNotFoundError(f"SARA cases directory not found at {cases_dir}")
        if not split_file.exists():
            raise FileNotFoundError(
                f"SARA split file not found at {split_file}. "
                "Set SARA_SPLIT=train or SARA_SPLIT=test."
            )

        case_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]

        cases: list[EvalCase] = []
        for case_id in case_ids:
            case_path = cases_dir / f"{case_id}.pl"
            if not case_path.exists():
                continue

            context_text, question, expected, expected_type = _parse_case_file(case_path)
            relevant_ids = extract_usc_refs(f"{context_text}\n{question}")

            cases.append(
                EvalCase(
                    id=case_id,
                    question=question,
                    rubric=expected or None,
                    relevant_ids=relevant_ids,
                    metadata={
                        "context": context_text,
                        "source_file": case_path.name,
                        "expected_type": expected_type,
                        "split": split,
                    },
                )
            )

            if limit is not None and len(cases) >= limit:
                break

        return cases

    def score(self, response: str, case: EvalCase, judge_fn) -> dict:
        """Score SARA responses with deterministic checks when possible.

        Numeric and entailment/contradiction cases are scored without API calls,
        with explicit checks for:
          - answer correctness,
          - citation grounding to section references present in case facts/question,
          - presence of multi-step numeric reasoning for arithmetic cases.

        Fallback is LLM-as-judge only when a freeform expected answer exists.
        """
        expected = (case.rubric or "").strip()
        expected_type = str(case.metadata.get("expected_type", "freeform"))
        citation = _citation_metrics_for_case(response or "", case)

        answer_correct = 0.0
        calc_steps_present = None

        if not expected:
            judge_result = {
                "earned": None,
                "total": None,
                "feedback": "No expected answer available.",
            }
        elif expected_type == "label":
            expected_label = expected.lower()
            response_label_match = _LABEL_RE.search(response or "")
            answer_correct = 1.0 if response_label_match and response_label_match.group(1).lower() == expected_label else 0.0

            # Label cases: prioritize correct legal conclusion, then citation grounding.
            earned = round(0.75 * answer_correct + 0.25 * citation["citation_fact_precision"], 4)
            judge_result = {
                "earned": earned,
                "total": 1.0,
                "feedback": (
                    f"Expected {expected}; "
                    f"answer_correct={bool(answer_correct)}; "
                    f"citation_precision={citation['citation_fact_precision']:.2f}"
                ),
            }
        elif expected_type == "numeric":
            expected_num = _extract_first_number(expected)
            response_numbers = _extract_all_numbers(response or "")
            answer_correct = 1.0 if expected_num and expected_num in response_numbers else 0.0

            calc_steps_present = bool(
                len(response_numbers) >= 2 and _CALC_MARKER_RE.search(response or "")
            )
            calc_score = 1.0 if calc_steps_present else 0.0

            # Numeric cases emphasize correct value, plus calculation trace and citation grounding.
            earned = round(
                0.60 * answer_correct
                + 0.20 * calc_score
                + 0.20 * citation["citation_fact_precision"],
                4,
            )
            judge_result = {
                "earned": earned,
                "total": 1.0,
                "feedback": (
                    f"Expected numeric {expected_num}; "
                    f"answer_correct={bool(answer_correct)}; "
                    f"calc_steps_present={bool(calc_steps_present)}; "
                    f"citation_precision={citation['citation_fact_precision']:.2f}"
                ),
            }
        else:
            rubric = f"[+1.00] Response matches expected SARA answer: {expected}"
            judge_result = judge_fn(response, rubric)
            answer_correct = float(judge_result.get("earned") or 0.0)

        judge_result["answer_correct"] = round(answer_correct, 4)
        judge_result["calculation_steps_present"] = calc_steps_present
        judge_result.update(citation)
        judge_result["citation_correct"] = citation["n_correct_cited_refs"] > 0

        rouge_result = _compute_rouge(response or "", expected) if expected else {}
        return {**judge_result, **rouge_result}