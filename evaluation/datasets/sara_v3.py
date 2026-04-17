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
from src.utils.reference_matching import best_match_score


_LABEL_RE = re.compile(r"\b(entailment|contradiction|unknown)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
_GOAL_NUMBER_RE = re.compile(r"(?<![A-Za-z_])-?[0-9]+(?:\.[0-9]+)?")
_CALC_MARKER_RE = re.compile(
    r"(?:step\s*\d+|=|\+|\-|\*|/|minus|plus|times|divide|therefore|final answer)",
    re.IGNORECASE,
)
_FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LABEL_ENTAILMENT_RE = re.compile(r"\b(entailment|entailed|entails)\b", re.IGNORECASE)
_LABEL_CONTRADICTION_RE = re.compile(
    r"\b(contradiction|contradictory|contradicts|contradict)\b",
    re.IGNORECASE,
)
_LABEL_UNKNOWN_RE = re.compile(
    r"\b(unknown|indeterminate|cannot determine|insufficient information)\b",
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
    numbers: list[str] = []
    for match in _NUMBER_RE.finditer(text or ""):
        start = match.start()
        lookback = (text[max(0, start - 16) : start]).lower()
        if "§" in lookback or lookback.rstrip().endswith("section"):
            continue
        numbers.append(_clean_money(match.group(1)))
    return numbers


def _extract_goal_numbers(goal_text: str) -> list[str]:
    return [_clean_money(token) for token in _GOAL_NUMBER_RE.findall(goal_text or "")]


def _is_likely_numeric_question(question_raw: str) -> bool:
    q = (question_raw or "").lower().strip()
    if not q:
        return False
    if "how much" in q:
        return True
    if "$" in q:
        return True
    if "tax" in q and "?" in q:
        return True
    if "amount" in q and "?" in q:
        return True
    return False


def _extract_final_answer(response: str) -> str | None:
    match = _FINAL_ANSWER_RE.search(response or "")
    if not match:
        return None
    answer = match.group(1).strip()
    if not answer:
        return None
    return answer.splitlines()[0].strip()


def _normalize_string(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _canonical_label_from_text(text: str, allow_boolean: bool = False) -> str | None:
    normalized = _normalize_string(text or "")
    if not normalized:
        return None

    if _LABEL_UNKNOWN_RE.search(normalized):
        return "unknown"
    if re.search(r"\b(not entailment|not entailed|does not entail)\b", normalized):
        return "contradiction"
    if re.search(r"\b(not contradiction|not contradictory)\b", normalized):
        return "entailment"
    if _LABEL_CONTRADICTION_RE.search(normalized):
        return "contradiction"
    if _LABEL_ENTAILMENT_RE.search(normalized):
        return "entailment"

    if allow_boolean:
        bool_candidate = normalized.strip()
        if bool_candidate in {"true", "yes"}:
            return "entailment"
        if bool_candidate in {"false", "no"}:
            return "contradiction"
        if bool_candidate.startswith("true ") or bool_candidate.startswith("yes "):
            return "entailment"
        if bool_candidate.startswith("false ") or bool_candidate.startswith("no "):
            return "contradiction"

    return None


def _extract_predicted_label(response: str) -> str | None:
    final_answer = _extract_final_answer(response or "")
    if final_answer:
        predicted = _canonical_label_from_text(final_answer, allow_boolean=True)
        if predicted:
            return predicted

    return _canonical_label_from_text(response or "", allow_boolean=False)


def _label_from_test_goal(case: EvalCase) -> str | None:
    test_goal = str(case.metadata.get("test_goal", "")).strip()
    if not test_goal:
        return None

    if bool(case.metadata.get("test_goal_negated")):
        return "contradiction"
    return "entailment"


def _label_from_case_stem(case_stem: str) -> str | None:
    if case_stem.endswith("_pos"):
        return "Entailment"
    if case_stem.endswith("_neg"):
        return "Contradiction"
    return None


_STATUTORY_PRED_RE = re.compile(r"^s[a-zA-Z0-9]+\s*\(", re.ASCII)


def _filter_statutory_facts(facts_text: str) -> str:
    """Return only statutory predicate lines from %Facts, dropping NLP span annotations.

    Keeps lines like: s151("Alice",2000,_,_,2017).
    Drops directives (:- ...) and SRL span predicates (payment_(span(...))).
    """
    kept: list[str] = []
    for line in facts_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(":-") or "span(" in stripped:
            continue
        if _STATUTORY_PRED_RE.match(stripped):
            kept.append(stripped)
    return "\n".join(kept)


def _split_prolog_clauses(lines: list[str]) -> list[str]:
    """Split Prolog lines into complete clauses ending in a period."""
    clauses: list[str] = []
    buffer: list[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        buffer.append(line)
        if line.endswith("."):
            clauses.append(" ".join(buffer))
            buffer = []

    if buffer:
        clauses.append(" ".join(buffer))

    return clauses


def _predicate_signature(term: str) -> tuple[str, int] | None:
    term = term.strip()
    if not term:
        return None

    match = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_]*)(?:\((.*)\))?", term)
    if not match:
        return None

    name = match.group(1)
    args = (match.group(2) or "").strip()
    if not args:
        return (name, 0)

    depth = 0
    arity = 1
    for ch in args:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        elif ch == "," and depth == 0:
            arity += 1

    return (name, arity)


def _parse_test_goals(test_lines: list[str]) -> dict:
    """Parse %Test clauses and infer asserted goal and effective negation."""
    clauses = _split_prolog_clauses(test_lines)
    if not clauses:
        return {
            "test_goal": "",
            "test_goal_negated": False,
            "test_goal_numbers": [],
        }

    definitions: dict[tuple[str, int], str] = {}
    queries: list[str] = []

    for clause in clauses:
        content = clause[:-1].strip() if clause.endswith(".") else clause.strip()
        if not content:
            continue

        if content.startswith(":-"):
            query_expr = content[2:].strip()
            if query_expr.lower() == "halt":
                continue
            queries.append(query_expr)
            continue

        if ":-" in content:
            head, body = content.split(":-", 1)
            signature = _predicate_signature(head.strip())
            if signature:
                definitions[signature] = body.strip()

    if not queries:
        return {
            "test_goal": "",
            "test_goal_negated": False,
            "test_goal_numbers": [],
        }

    primary_query = queries[0]
    query_negated = primary_query.startswith("\\+")
    query_term = primary_query[2:].strip() if query_negated else primary_query

    signature = _predicate_signature(query_term)
    if signature and signature in definitions:
        resolved_body = definitions[signature]
        if resolved_body.startswith("\\+"):
            query_negated = True

    return {
        "test_goal": query_term,
        "test_goal_negated": query_negated,
        "test_goal_numbers": _extract_goal_numbers(query_term),
    }


def _citation_metrics_for_case(response: str, case: EvalCase) -> dict:
    """Compute citation correctness against SARA case fact/question references."""
    cited_refs = extract_usc_refs(response or "")
    cited_lower = [ref.lower() for ref in cited_refs]

    fact_refs = [ref.lower() for ref in case.relevant_ids]
    fact_ref_set = set(fact_refs)

    # Raw exact matching (kept for diagnostics).
    matched_exact = [ref for ref in cited_refs if ref.lower() in fact_ref_set]
    matched_exact_set = {ref.lower() for ref in matched_exact}

    # Normalized matching: map each citation to the best fact reference and
    # collapse over-citation within the same fact branch.
    normalized_keys: set[str] = set()
    matched_fact_set: set[str] = set()
    for citation in cited_lower:
        score, best_fact = best_match_score(citation, list(fact_ref_set))
        if score > 0 and best_fact is not None:
            normalized_keys.add(f"fact:{best_fact}")
            matched_fact_set.add(best_fact)
        else:
            normalized_keys.add(f"raw:{citation}")

    normalized_total = len(normalized_keys)
    normalized_matched = len({k for k in normalized_keys if k.startswith("fact:")})

    if not fact_ref_set:
        precision = 1.0 if not cited_refs else 0.0
        recall = None
        precision_raw = precision
        recall_raw = recall
    else:
        precision = (normalized_matched / normalized_total) if normalized_total else 0.0
        recall = len(matched_fact_set) / len(fact_ref_set)
        precision_raw = (len(matched_exact) / len(cited_refs)) if cited_refs else 0.0
        recall_raw = len(matched_exact_set) / len(fact_ref_set)

    return {
        "cited_refs": cited_refs,
        "matched_refs": sorted(matched_fact_set),
        "citation_fact_precision": round(precision, 4),
        "citation_fact_recall": round(recall, 4) if recall is not None else None,
        "citation_fact_precision_raw": round(precision_raw, 4),
        "citation_fact_recall_raw": round(recall_raw, 4) if recall_raw is not None else None,
        "n_fact_refs": len(fact_ref_set),
        "n_cited_refs": len(cited_refs),
        "n_correct_cited_refs": len(matched_fact_set),
        "n_cited_refs_normalized": normalized_total,
    }


def _parse_case_file(path: Path) -> dict:
    """Parse one SARA case file.

    Returns:
      Dict with context/facts/question/expected answer fields.
    expected_type is one of: label, numeric, string, freeform.
    """
    text_lines: list[str] = []
    question_lines: list[str] = []
    facts_lines: list[str] = []
    test_lines: list[str] = []
    section: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if line.startswith("% Text"):
            section = "text"
            continue
        if line.startswith("% Question"):
            section = "question"
            continue
        if line.startswith("% Facts"):
            section = "facts"
            continue
        if line.startswith("% Test"):
            section = "test"
            continue

        if section in {"text", "question"}:
            if not line.startswith("%"):
                continue
            content = line[1:].strip()
            if not content:
                continue
            if section == "text":
                text_lines.append(content)
            else:
                question_lines.append(content)
            continue

        if section == "facts":
            content = line.strip()
            if content:
                facts_lines.append(content)
            continue

        if section == "test":
            content = line.strip()
            if content:
                test_lines.append(content)

    context_text = " ".join(text_lines).strip()
    facts_text = _filter_statutory_facts("\n".join(facts_lines))
    question_raw = " ".join(question_lines).strip()

    expected = ""
    expected_type = "freeform"
    question = question_raw
    label_from_stem = _label_from_case_stem(path.stem)
    label_from_question: str | None = None

    if "?" in question_raw:
        before, after = question_raw.rsplit("?", 1)
        question = before.strip() + "?"
        candidate = after.strip()
        if candidate:
            expected = candidate

    label_match = _LABEL_RE.search(question_raw)
    if label_match:
        label_from_question = label_match.group(1).capitalize()
        if not expected:
            expected = label_from_question
        question = question_raw[: label_match.start()].strip().rstrip(".")
        if question and not question.endswith("?"):
            question += "?"

    test_goal_info = _parse_test_goals(test_lines)
    primary_test_goal = test_goal_info["test_goal"]
    primary_test_goal_negated = bool(test_goal_info["test_goal_negated"])
    primary_test_goal_numbers = list(test_goal_info["test_goal_numbers"])

    expected_number = _extract_first_number(expected or "")
    test_number = primary_test_goal_numbers[-1] if primary_test_goal_numbers else None

    # Use %Test numeric answer as ground truth when question extraction is
    # missing or inconsistent.
    if test_number and (
        (expected_number is not None and expected_number != test_number)
        or (not expected and _is_likely_numeric_question(question_raw))
    ):
        expected = test_number

    if not expected:
        is_interrogative = "?" in question_raw

        if label_from_question:
            expected = label_from_question
        elif not is_interrogative and (label_from_stem or primary_test_goal):
            if primary_test_goal_negated:
                expected = "Contradiction"
            else:
                expected = label_from_stem or "Entailment"
        elif primary_test_goal_numbers:
            expected = primary_test_goal_numbers[-1]
        elif label_from_stem:
            expected = label_from_stem

    if _LABEL_RE.fullmatch(expected.strip()):
        expected_type = "label"
    elif _extract_first_number(expected or "") is not None:
        expected_type = "numeric"
    elif expected:
        expected_type = "string"

    if not question:
        question = question_raw

    return {
        "context_text": context_text,
        "facts_text": facts_text,
        "question": question,
        "expected": expected,
        "expected_type": expected_type,
        "test_goal": primary_test_goal,
        "test_goal_negated": primary_test_goal_negated,
        "test_goal_numbers": primary_test_goal_numbers,
    }


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


_SECTION_KEY_RE = re.compile(r"^(s\d+|tax_case)", re.IGNORECASE)

# Map case-id prefix to the statute source filenames that are authoritative for it.
# tax_case_* needs all four sections for full tax calculation.
_CASE_PREFIX_TO_STATUTES: dict[str, list[str]] = {
    "s1_":    ["section1"],
    "s2_":    ["section2"],
    "s63_":   ["section63"],
    "s68_":   ["section68"],
    "s151_":  ["section151"],
    "s152_":  ["section152"],
    "s3301_": ["section3301"],
    "s3306_": ["section3306"],
    "s7703_": ["section7703"],
    "tax_case_": ["section1", "section2", "section63", "section68", "section151", "section3301", "section3306"],
}


_STATUTE_FILE_RE = re.compile(r"^section([0-9]{1,4}[a-z]?)$", re.IGNORECASE)


def _statute_files_for_case(case_id: str) -> list[str]:
    for prefix, filenames in _CASE_PREFIX_TO_STATUTES.items():
        if case_id.startswith(prefix):
            return list(filenames)
    return []


def _refs_from_statute_files(filenames: list[str]) -> list[str]:
    refs: set[str] = set()
    for filename in filenames:
        match = _STATUTE_FILE_RE.fullmatch(filename.strip())
        if not match:
            continue
        refs.add(f"26 USC §{match.group(1).lower()}")
    return sorted(refs)


def _load_sara_statutes(case_id: str, statutes_dir: Path) -> str:
    """Return the concatenated text of SARA source statutes relevant to case_id."""
    filenames = _statute_files_for_case(case_id)
    if filenames:
        parts: list[str] = []
        for fname in filenames:
            fpath = statutes_dir / fname
            if fpath.exists():
                parts.append(fpath.read_text(encoding="utf-8").strip())
        return "\n\n---\n\n".join(parts)
    return ""


def _derive_sara_references(
    case_id: str,
    context_text: str,
    facts_text: str,
    question: str,
    sara_statutes: str,
) -> tuple[list[str], list[str]]:
    """Return (allowed_refs, relevant_ids) for SARA evaluation.

    - allowed_refs: concise list used in the prompt citation constraint.
    - relevant_ids: broader list used for retrieval/citation scoring.
    """
    statute_files = _statute_files_for_case(case_id)
    allowed_refs = _refs_from_statute_files(statute_files)

    relevant_ids = set(
        extract_usc_refs(f"{context_text}\n{facts_text}\n{question}\n{sara_statutes}")
    )
    relevant_ids.update(allowed_refs)

    # Fallback for edge cases where parsing fails to detect any USC refs.
    if not relevant_ids:
        relevant_ids.update(allowed_refs)

    return (sorted(allowed_refs), sorted(relevant_ids))


class SARAV3Dataset(Dataset):
    """Adapter for dataset/sara_v3."""

    name = "sara_v3"

    def load(self, data_dir: Path, limit: int | None = None) -> list[EvalCase]:
        root = data_dir / "sara_v3"
        cases_dir = root / "cases"
        splits_dir = root / "splits"
        statutes_dir = root / "statutes" / "source"

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

            parsed = _parse_case_file(case_path)
            context_text = parsed["context_text"]
            facts_text = parsed["facts_text"]
            question = parsed["question"]
            expected = parsed["expected"]
            expected_type = parsed["expected_type"]

            sara_statutes = _load_sara_statutes(case_id, statutes_dir) if statutes_dir.exists() else ""
            allowed_refs, relevant_ids = _derive_sara_references(
                case_id=case_id,
                context_text=context_text,
                facts_text=facts_text,
                question=question,
                sara_statutes=sara_statutes,
            )

            cases.append(
                EvalCase(
                    id=case_id,
                    question=question,
                    rubric=expected or None,
                    relevant_ids=relevant_ids,
                    metadata={
                        "context": context_text,
                        "facts": facts_text,
                        "source_file": case_path.name,
                        "expected_type": expected_type,
                        "test_goal": parsed["test_goal"],
                        "test_goal_negated": parsed["test_goal_negated"],
                        "test_goal_numbers": parsed["test_goal_numbers"],
                        "allowed_refs": allowed_refs,
                        "split": split,
                        "sara_statutes": sara_statutes,
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
        final_answer = _extract_final_answer(response or "")

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
            predicted_label = _extract_predicted_label(response or "")
            answer_correct = 1.0 if predicted_label == expected_label else 0.0
            test_goal_label = _label_from_test_goal(case)
            expected_label_consistent_with_test = None
            if test_goal_label is not None:
                expected_label_consistent_with_test = expected_label == test_goal_label

            # Label cases: prioritize correct legal conclusion, then citation grounding.
            earned = round(0.75 * answer_correct + 0.25 * citation["citation_fact_precision"], 4)
            judge_result = {
                "earned": earned,
                "total": 1.0,
                "feedback": (
                    f"Expected {expected}; "
                    f"predicted_label={predicted_label or 'none'}; "
                    f"answer_correct={bool(answer_correct)}; "
                    f"citation_precision={citation['citation_fact_precision']:.2f}"
                ),
                "predicted_label": predicted_label,
                "test_goal_label": test_goal_label,
                "expected_label_consistent_with_test": expected_label_consistent_with_test,
            }
        elif expected_type == "numeric":
            expected_num = _extract_first_number(expected)
            number_source = final_answer if final_answer else (response or "")
            response_numbers = _extract_all_numbers(number_source)
            answer_correct = 1.0 if expected_num and expected_num in response_numbers else 0.0

            calc_steps_present = bool(
                len(_extract_all_numbers(response or "")) >= 2
                and _CALC_MARKER_RE.search(response or "")
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
        elif expected_type == "string":
            expected_norm = _normalize_string(expected)
            response_norm = _normalize_string(final_answer or (response or ""))
            answer_correct = 1.0 if expected_norm and expected_norm in response_norm else 0.0

            earned = round(0.75 * answer_correct + 0.25 * citation["citation_fact_precision"], 4)
            judge_result = {
                "earned": earned,
                "total": 1.0,
                "feedback": (
                    f"Expected '{expected}'; "
                    f"answer_correct={bool(answer_correct)}; "
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