"""Run batch evaluation against a tax Q&A benchmark.

Usage:
  # Single dataset, single mode
  python evaluation/run_eval.py --dataset taxbench --mode hybrid --model claude

  # All four retrieval modes back-to-back
  python evaluation/run_eval.py --dataset taxbench --mode all --model claude

  # Dry run (no API calls; useful for testing the pipeline)
  python evaluation/run_eval.py --dataset taxbench --mode hybrid --dry-run

  # Limit to first N cases
  python evaluation/run_eval.py --dataset taxbench --mode hybrid --limit 10

Results are saved to evaluation/results/<dataset>__<model>__<mode>.json.
Run evaluation/score_results.py afterward to print the summary table.

Adding a new dataset
--------------------
1. Create evaluation/datasets/<name>.py implementing Dataset (see base.py).
2. Import it and add it to REGISTRY in evaluation/datasets/__init__.py.
3. Pass --dataset <name> to this script.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.datasets import REGISTRY
from evaluation.datasets.base import Dataset, EvalCase
from src import config as cfg
from src.ingestion.irs_xml_parser import SOURCE_LABELS
from src.preprocessing.normalizer import extract_irs_refs, extract_usc_refs
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.reference_matching import best_match_score


# ---------------------------------------------------------------------------
# Retrieval modes (maps to the four experimental conditions)
# ---------------------------------------------------------------------------

ALL_MODES = ["none", "vector", "graph", "hybrid"]

# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are grading a tax law Q&A response against a scoring rubric.

Rubric (each [+N.NN] is one criterion with that many points):
{rubric}

Response to grade:
{response}

For each criterion in the rubric, decide whether the response satisfies it.
Return a JSON object with exactly these fields:
  "earned" - float: total points earned (sum of criteria the response satisfies)
  "total"  - float: total points available (sum of all criterion weights)
  "feedback" - string: one or two sentences explaining what was correct or missing

Return only valid JSON, no extra text."""


def judge_response(response: str, rubric: str, model_id: str) -> dict:
    """Ask a Claude or Gemini model to score a response against a rubric."""
    prompt = _JUDGE_PROMPT.format(rubric=rubric, response=response)

    if model_id.startswith("claude"):
        return _judge_claude(prompt, model_id)
    if model_id.startswith("gemini"):
        return _judge_gemini(prompt, model_id)
    if model_id.startswith("ollama:"):
        return _judge_ollama(prompt, model_id)
    raise ValueError(f"Unsupported judge model: {model_id!r}")


def _judge_claude(prompt: str, model_id: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=model_id,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    return _parse_judge_json(raw)


def _judge_gemini(prompt: str, model_id: str) -> dict:
    from google import genai
    client = genai.Client(api_key=cfg.GEMINI_API_KEY)
    response = client.models.generate_content(model=model_id, contents=prompt)
    raw = response.text.strip()
    return _parse_judge_json(raw)


def _judge_ollama(prompt: str, model_id: str) -> dict:
    raw = _call_ollama(
        "Return only valid JSON matching the user's requested schema.",
        prompt,
        model_id,
    )
    return _parse_judge_json(raw)


def _parse_judge_json(raw: str) -> dict:
    """Parse JSON from judge response; fall back gracefully.

    On parse failure, earned/total are set to None (not 0.0) so that
    _print_summary can exclude these cases from score averages and report
    them separately rather than silently pulling the mean toward zero.
    """
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        result = json.loads(raw)
        return {
            "earned":   float(result.get("earned", 0.0)),
            "total":    float(result.get("total", 0.0)),
            "feedback": str(result.get("feedback", "")),
        }
    except (json.JSONDecodeError, ValueError, KeyError):
        return {
            "earned":      None,
            "total":       None,
            "feedback":    f"Judge response could not be parsed: {raw[:200]}",
            "_parse_error": True,
        }


# ---------------------------------------------------------------------------
# LLM response generation
# ---------------------------------------------------------------------------

_SYSTEM_WITH_CONTEXT = """\
You are a knowledgeable U.S. federal tax assistant.

For every answer you must:
1. Identify the relevant legal rules and cite each statute or IRS publication \
used (e.g., "Under 26 USC §63...", "Per IRS Pub. 17...").
2. Explain how those rules apply to the specific facts or question.
3. For numeric questions, show full step-by-step arithmetic before the answer.

{context_block}"""

_SYSTEM_NO_CONTEXT = """\
You are a knowledgeable U.S. federal tax assistant answering from training knowledge.

For every answer you must:
1. Identify the relevant legal rules and cite each statute or IRS publication used.
2. Explain how those rules apply to the specific facts or question.
3. For numeric questions, show full step-by-step arithmetic before the answer."""

_SARA_SYSTEM_APPEND = """\

SARA output format — required for all answer types:

Step 1 — Legal Rule: State the exact condition the cited statute imposes.
Step 2 — Facts Applied: Map each case fact to that condition — which facts satisfy it, which do not.
Step 3 — Reasoning: Explain why the facts do or do not satisfy the rule.
Final Answer: <value-or-label>

Constraints:
- The Final Answer line is mandatory — write it even if the steps above are incomplete.
- Use ONLY the provided case facts for taxpayer-specific values.
- Treat "SARA Authoritative Statute Text" as the primary legal source for ALL thresholds, rates, and rules.
- Retrieved excerpts provide additional detail only; if they conflict with SARA statute text, follow SARA statute text.
- When an allowed citation list is provided, cite from that list.
"""


def _sara_label_options(case: EvalCase) -> list[str]:
    """Return allowed label outputs for a SARA label case.

    SARA v3 is predominantly a binary entailment/contradiction task.
    Allow Unknown only when the question text explicitly signals an
    indeterminate label scenario.
    """
    question = str(case.question or "").lower()
    allow_unknown = bool(
        re.search(
            r"\b(unknown|indeterminate|cannot\s+determine|insufficient\s+information)\b",
            question,
        )
    )
    options = ["Entailment", "Contradiction"]
    if allow_unknown:
        options.append("Unknown")
    return options


def build_system_prompt(chunks: list[dict]) -> str:
    if not chunks:
        return _SYSTEM_NO_CONTEXT
    excerpts = [
        f"[{c['section_id']}]\n{_clip_excerpt(str(c.get('text', '')), cfg.PROMPT_EXCERPT_MAX_CHARS)}"
        for c in chunks
    ]
    context_block = (
        "Relevant statutory and IRS guidance excerpts:\n\n"
        + "\n\n---\n\n".join(excerpts)
    )
    return _SYSTEM_WITH_CONTEXT.format(context_block=context_block)


def _clip_excerpt(text: str, max_chars: int) -> str:
    """Clip long context excerpts to reduce prompt noise and latency."""
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    clipped = text[: max(0, max_chars - 3)].rstrip()
    return f"{clipped}..."


def _build_sara_user_prompt(case: EvalCase) -> str:
    """Build SARA case prompt with explicit facts, citation constraints, and reasoning structure."""
    text_context = str(case.metadata.get("context", "")).strip()
    structured_facts = str(case.metadata.get("facts", "")).strip()
    expected_type = str(case.metadata.get("expected_type", "freeform")).strip().lower()
    allowed_refs = list(case.metadata.get("allowed_refs") or case.relevant_ids or [])
    if allowed_refs:
        shown = allowed_refs[:24]
        if len(allowed_refs) > 24:
            allowed_text = ", ".join(shown) + f", ... (+{len(allowed_refs) - 24} more)"
        else:
            allowed_text = ", ".join(shown)
    else:
        allowed_text = "(not provided; cite section references from the SARA statute text)"

    if expected_type == "label":
        label_options = _sara_label_options(case)
        label_choices = " / ".join(label_options)
        type_instructions = (
            f"This is a legal entailment task.\n"
            f"Step 1 — Legal Rule: What exact condition does the cited law impose?\n"
            f"Step 2 — Facts Applied: Do the case facts satisfy every condition?\n"
            f"Step 3 — Reasoning: Why does this lead to entailment or contradiction?\n"
            f"Final Answer must be exactly one of: {label_choices}"
        )
    elif expected_type == "numeric":
        type_instructions = (
            "This is a numeric calculation task.\n"
            "IMPORTANT: Use ONLY the dollar amounts, rates, and thresholds from the "
            "SARA Authoritative Statute Text provided above — do NOT use tax rates "
            "from your training knowledge.\n"
            "Step 1 — Legal Rule: Copy the exact formula or bracket table from the statute text above.\n"
            "Step 2 — Facts Applied: Substitute each case value into the formula.\n"
            "Step 3 — Arithmetic: Show every calculation step with actual numbers.\n"
            "Final Answer must be a number (digits only, no $ sign, no commas)."
        )
    elif expected_type == "string":
        type_instructions = (
            "This is a factual lookup task.\n"
            "Step 1 — Legal Rule: What does the cited law define or provide?\n"
            "Step 2 — Facts Applied: Which case details are relevant?\n"
            "Step 3 — Reasoning: How do you arrive at the answer?\n"
            "Final Answer must be the specific value."
        )
    else:
        type_instructions = (
            "Step 1 — Legal Rule: Identify the relevant rule.\n"
            "Step 2 — Facts Applied: Apply it to the case facts.\n"
            "Step 3 — Reasoning: Explain the conclusion.\n"
            "Final Answer must be present."
        )

    facts_block = (
        "Statutory facts in Prolog notation — s<section>(<person>, <amount/status>, ..., <year>).\n"
        "Example: s63(\"Alice\",2017,26948) means Alice's §63 taxable income in 2017 is $26,948.\n"
        f"{structured_facts}"
        if structured_facts
        else "(no structured facts provided)"
    )

    sara_statutes = str(case.metadata.get("sara_statutes", "")).strip()
    statute_block = (
        "\nSARA Authoritative Statute Text (use these exact thresholds and formulas):\n"
        f"{sara_statutes}\n"
        if sara_statutes
        else ""
    )

    return (
        "SARA Case Text (%Text):\n"
        f"{text_context or '(no text context provided)'}\n\n"
        "SARA Structured Facts (%Facts):\n"
        f"{facts_block}\n"
        f"{statute_block}\n"
        "Question:\n"
        f"{case.question}\n\n"
        "Task instructions:\n"
        f"{type_instructions}\n"
        "Remember: Output strictly the Step 1, Step 2, Step 3, and your Final Answer format. Use only SARA case facts and SARA rules provided here."
    )


def _build_sara_system_append(case: EvalCase) -> str:
    """Return a case-type-specific system append reinforcing the required output format."""
    expected_type = str(case.metadata.get("expected_type", "freeform")).strip().lower()

    if expected_type == "label":
        label_options = _sara_label_options(case)
        label_choices = " / ".join(label_options)
        return (
            f"{_SARA_SYSTEM_APPEND}"
            f"- Final Answer for this label task must be exactly one of: {label_choices}\n"
            "- Do not output a number for a label task.\n"
        )
    if expected_type == "numeric":
        return (
            f"{_SARA_SYSTEM_APPEND}"
            "- Show all arithmetic steps before the Final Answer.\n"
            "- Final Answer must be a number (e.g., 'Final Answer: 4200').\n"
        )
    return _SARA_SYSTEM_APPEND


def call_llm(system: str, question: str, model_id: str, max_tokens: int | None = None) -> str:
    """Call the configured LLM and return the response text."""
    if model_id.startswith("claude"):
        return _call_claude(system, question, model_id, max_tokens)
    if model_id.startswith("gemini"):
        return _call_gemini(system, question, model_id, max_tokens)
    if model_id.startswith("ollama:"):
        return _call_ollama(system, question, model_id, max_tokens)
    raise ValueError(f"Unsupported model: {model_id!r}")


def _call_claude(system: str, question: str, model_id: str, max_tokens: int | None = None) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=model_id,
        max_tokens=max_tokens or 2048,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return message.content[0].text


def _call_gemini(system: str, question: str, model_id: str, max_tokens: int | None = None) -> str:
    from google import genai
    client = genai.Client(api_key=cfg.GEMINI_API_KEY)
    from google.genai import types
    response = client.models.generate_content(
        model=model_id,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens or 2048,
        ),
    )
    return response.text


def _call_ollama(system: str, question: str, model_id: str, max_tokens: int | None = None) -> str:
    model_name = model_id.split("ollama:", 1)[1] if model_id.startswith("ollama:") else model_id
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "think": cfg.OLLAMA_THINK,
    }

    options: dict[str, int | float] = {}
    if cfg.OLLAMA_NUM_CTX > 0:
        options["num_ctx"] = int(cfg.OLLAMA_NUM_CTX)
    # max_tokens (per-call override) takes priority over the global OLLAMA_NUM_PREDICT.
    effective_predict = max_tokens or (cfg.OLLAMA_NUM_PREDICT if cfg.OLLAMA_NUM_PREDICT != 0 else None)
    if effective_predict is not None:
        options["num_predict"] = int(effective_predict)
    if cfg.OLLAMA_TEMPERATURE >= 0:
        options["temperature"] = float(cfg.OLLAMA_TEMPERATURE)
    if 0.0 < cfg.OLLAMA_TOP_P <= 1.0:
        options["top_p"] = float(cfg.OLLAMA_TOP_P)
    # Push model layers to GPU. OLLAMA_NUM_GPU=99 forces all layers onto GPU;
    # Ollama caps at the model's actual layer count so 99 is always safe.
    if cfg.OLLAMA_NUM_GPU >= 0:
        options["num_gpu"] = int(cfg.OLLAMA_NUM_GPU)
    if options:
        payload["options"] = options

    body = json.dumps(payload).encode("utf-8")
    endpoint = cfg.OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
    req = urlrequest.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    retries = max(1, int(cfg.OLLAMA_MAX_RETRIES))
    backoff = max(0, int(cfg.OLLAMA_RETRY_BACKOFF_SECONDS))

    for attempt in range(1, retries + 1):
        try:
            with urlrequest.urlopen(req, timeout=cfg.OLLAMA_TIMEOUT_SECONDS) as resp:
                raw = resp.read().decode("utf-8")

            data = json.loads(raw)
            message = data.get("message", {})
            content = str(message.get("content", "")).strip()
            if not content:
                raise RuntimeError(f"Ollama returned empty response: {raw[:200]}")
            return content

        except urlerror.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""

            if exc.code == 404:
                raise RuntimeError(
                    f"Ollama model '{model_name}' was not found. "
                    f"Run: ollama pull {model_name}."
                ) from exc

            raise RuntimeError(
                f"Ollama HTTP error {exc.code}. "
                f"Endpoint={cfg.OLLAMA_BASE_URL}. Details={detail[:200]}"
            ) from exc

        except (TimeoutError, socket.timeout) as exc:
            if attempt < retries:
                print(
                    f"    Ollama timeout (attempt {attempt}/{retries}); retrying in {backoff}s..."
                )
                if backoff > 0:
                    time.sleep(backoff)
                continue
            raise RuntimeError(
                "Ollama request timed out. "
                f"Model={model_name}, timeout={cfg.OLLAMA_TIMEOUT_SECONDS}s, retries={retries}. "
                "Try increasing OLLAMA_TIMEOUT_SECONDS, reducing --limit, or using a smaller model."
            ) from exc

        except urlerror.URLError as exc:
            reason = getattr(exc, "reason", None)
            is_timeout = isinstance(reason, (TimeoutError, socket.timeout))
            if is_timeout and attempt < retries:
                print(
                    f"    Ollama timeout (attempt {attempt}/{retries}); retrying in {backoff}s..."
                )
                if backoff > 0:
                    time.sleep(backoff)
                continue

            if is_timeout:
                raise RuntimeError(
                    "Ollama request timed out. "
                    f"Model={model_name}, timeout={cfg.OLLAMA_TIMEOUT_SECONDS}s, retries={retries}. "
                    "Try increasing OLLAMA_TIMEOUT_SECONDS, reducing --limit, or using a smaller model."
                ) from exc

            raise RuntimeError(
                "Failed to reach Ollama server. "
                f"Check OLLAMA_BASE_URL ({cfg.OLLAMA_BASE_URL}) and run 'ollama serve'."
            ) from exc

    raise RuntimeError("Ollama request failed unexpectedly.")


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

_LABEL_TO_SOURCE = {label.lower(): source for source, label in SOURCE_LABELS.items()}


def _source_from_reference(ref: str) -> str | None:
    ref_lower = ref.lower().strip()
    if ref_lower.startswith("26 usc §"):
        return "usc26"

    for label, source in _LABEL_TO_SOURCE.items():
        if ref_lower.startswith(label):
            return source
    return None


def _split_sentences(text: str) -> list[str]:
    return [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", text) if seg.strip()]


def _extract_response_citations(text: str) -> list[str]:
    refs = set(extract_usc_refs(text) + extract_irs_refs(text))
    return sorted(refs)


def _is_citation_supported(ref_lower: str, retrieved_lower: set[str]) -> bool:
    """Return True if ref is grounded by any retrieved section.

    Supports hierarchical matching: a cited parent section (e.g. "26 usc §32")
    is considered supported when a subsection (e.g. "26 usc §32(c)") was retrieved.
    """
    if ref_lower in retrieved_lower:
        return True
    return any(
        rid == ref_lower or rid.startswith(ref_lower + "(")
        for rid in retrieved_lower
    )


def _compute_citation_metrics(response: str, retrieved_ids: list[str]) -> dict:
    """Compute citation precision and grounding-rate style metrics.

    citation_precision = supported_citations / cited_citations
    grounding_rate     = citation_sentences_with_support / citation_sentences
    """
    retrieved_lower = {rid.lower() for rid in retrieved_ids}
    citations = _extract_response_citations(response)
    supported = [ref for ref in citations if _is_citation_supported(ref.lower(), retrieved_lower)]

    citation_precision = round(len(supported) / len(citations), 4) if citations else None

    citation_sentences = 0
    grounded_sentences = 0
    for sentence in _split_sentences(response):
        sentence_refs = _extract_response_citations(sentence)
        if not sentence_refs:
            continue
        citation_sentences += 1
        if any(_is_citation_supported(ref.lower(), retrieved_lower) for ref in sentence_refs):
            grounded_sentences += 1

    grounding_rate = round(grounded_sentences / citation_sentences, 4) if citation_sentences > 0 else None

    return {
        "n_citations": len(citations),
        "n_supported_citations": len(supported),
        "citation_precision": citation_precision,
        "citation_sentences": citation_sentences,
        "grounded_sentences": grounded_sentences,
        "grounding_rate": grounding_rate,
        "citations": citations,
        "supported_citations": sorted(supported),
    }


def _compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    retrieved_sources: list[str],
) -> dict:
    """Compute recall@k and MRR given retrieved and relevant section IDs.

    Args:
        retrieved_ids: Ordered list of retrieved section IDs (best first).
        relevant_ids:  Ground-truth relevant section IDs (unordered).

    Returns:
        Dict with recall_at_k and mrr (both floats in [0, 1]).
        Empty dict if relevant_ids is empty.
    """
    if not relevant_ids or not retrieved_ids:
        return {}

    relevant_set = {r.lower() for r in relevant_ids}
    retrieved_lower = [r.lower() for r in retrieved_ids]

    found_exact = sum(1 for rid in retrieved_lower if rid in relevant_set)
    recall_exact = found_exact / len(relevant_set)

    # Relaxed hierarchical recall: partial credit for parent/child subsection matches.
    best_scores = []
    for relevant in relevant_set:
        best_score, _ = best_match_score(relevant, retrieved_lower)
        best_scores.append(best_score)
    recall_hier = sum(best_scores) / len(relevant_set)

    mrr_exact = 0.0
    for rank, rid in enumerate(retrieved_lower, start=1):
        if rid in relevant_set:
            mrr_exact = 1.0 / rank
            break

    # Weighted reciprocal rank with hierarchical match scores.
    mrr_hier = 0.0
    for rank, rid in enumerate(retrieved_lower, start=1):
        score, _ = best_match_score(rid, list(relevant_set))
        if score > 0:
            mrr_hier = max(mrr_hier, score / rank)

    result = {
        # Backward-compatible keys now report relaxed hierarchical metrics.
        "recall_at_k": round(recall_hier, 4),
        "mrr": round(mrr_hier, 4),
        # Exact metrics remain available for strict evaluation comparisons.
        "recall_at_k_exact": round(recall_exact, 4),
        "mrr_exact": round(mrr_exact, 4),
    }

    relevant_sources = {
        source
        for source in (_source_from_reference(ref) for ref in relevant_ids)
        if source
    }
    retrieved_source_set = {source for source in retrieved_sources if source}
    if relevant_sources:
        source_hits = {
            source: (1.0 if source in retrieved_source_set else 0.0)
            for source in sorted(relevant_sources)
        }
        source_recall = sum(source_hits.values()) / len(source_hits)
        result["source_recall"] = round(source_recall, 4)
        result["source_hits"] = source_hits

    return result


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------

_MOCK_RESPONSE = (
    "Dry-run response. No API call was made. "
    "Re-run without --dry-run for real model output."
)


def _chunk_relevance_score(chunk: dict) -> float:
    """Return a unified relevance score across retriever modes."""
    for key in ("score", "vector_score", "graph_score"):
        value = chunk.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _dedupe_chunks_by_section(chunks: list[dict]) -> list[dict]:
    """Keep one highest-scoring chunk per section_id to reduce prompt redundancy."""
    by_section: dict[str, dict] = {}

    for chunk in chunks:
        section_id = str(chunk.get("section_id", "")).strip()
        if not section_id:
            continue

        current = by_section.get(section_id)
        if current is None or _chunk_relevance_score(chunk) > _chunk_relevance_score(current):
            by_section[section_id] = chunk

    ranked = sorted(by_section.values(), key=_chunk_relevance_score, reverse=True)
    return ranked


def _sara_ref_matches_section(section_id: str, allowed_refs: list[str]) -> bool:
    sid = (section_id or "").strip().lower()
    if not sid or not allowed_refs:
        return False
    for ref in allowed_refs:
        canonical = str(ref or "").strip().lower()
        if not canonical:
            continue
        if sid == canonical or sid.startswith(canonical + "("):
            return True
    return False


def _prioritize_sara_supplement_chunks(
    chunks: list[dict], case: EvalCase, mode: str = "vector"
) -> list[dict]:
    """Rank SARA retrieval chunks with statute-first, supplement-second policy.

    When SARA_SUPPLEMENT_ALLOWED_REFS_ONLY is True (default), only chunks whose
    section_id matches the case allowed_refs are kept. This prevents unrelated
    USC26 text or IRS publication fragments from competing with the authoritative
    SARA statute text already present in the user prompt, which would confuse
    small models and reduce answer accuracy.

    For hybrid mode, an additional intersection filter is applied: only chunks
    that appeared in BOTH the vector and graph retrieval paths are kept. This
    removes the parent-vs-child redundancy (e.g., §63 AND §63(c)(7) both being
    injected) that overwhelms small-model attention. The cap is also tighter.
    """
    allowed_refs = list(case.metadata.get("allowed_refs") or case.relevant_ids or [])

    # For hybrid: only keep chunks confirmed by BOTH retrieval paths (intersection).
    # This removes parent/child redundancy (§63 AND §63(c)(7) from two indexes)
    # that overwhelms small models. Fallback to all chunks if intersection is empty.
    if mode == "hybrid":
        consensus = [
            c for c in chunks
            if float(c.get("vector_score", 0.0)) > 0 and float(c.get("graph_score", 0.0)) > 0
        ]
        if consensus:
            chunks = consensus

    # Mode-aware cap:
    # - graph/hybrid: cap at 2 chunks to avoid flooding the 2B model with
    #   multiple overlapping subsection fragments (§63, §63(c), §63(c)(7) all
    #   passing the allowed_refs filter creates redundant competing context).
    # - vector: allow up to SARA_SUPPLEMENT_TOP_K (semantic search returns
    #   diverse paragraphs rather than overlapping subsection hierarchy).
    top_k = 2 if mode in ("graph", "hybrid") else cfg.SARA_SUPPLEMENT_TOP_K

    ranked: list[tuple[int, int, float, str, dict]] = []

    for chunk in chunks:
        section_id = str(chunk.get("section_id", "")).strip()
        source = str(chunk.get("source", "")).strip().lower()

        if (
            not cfg.SARA_SUPPLEMENT_ALLOW_GRAPH_COMMUNITY
            and section_id.lower().startswith("community:")
        ):
            continue

        is_allowed_ref_match = _sara_ref_matches_section(section_id, allowed_refs)

        # When strict mode is on, skip chunks that don't match an allowed ref.
        if cfg.SARA_SUPPLEMENT_ALLOWED_REFS_ONLY and not is_allowed_ref_match:
            continue

        if cfg.SARA_SUPPLEMENT_PRIORITIZE_ALLOWED_REFS and is_allowed_ref_match:
            tier = 0
            # Within tier-0, prefer shallower (parent) sections over deep subsections.
            # §63 (depth 0) ranks above §63(c)(7) (depth 2). The full authoritative
            # statute section covers the subsections already; adding deep subsections
            # creates overlapping context that confuses small models.
            section_depth = section_id.count("(")
        elif source == "usc26":
            tier = 1
            section_depth = 0
        elif source and source != "graph_community":
            tier = 2
            section_depth = 0
        else:
            tier = 3
            section_depth = 0

        # Sort: tier ASC, section_depth ASC, relevance DESC, section_id ASC.
        ranked.append((tier, section_depth, -_chunk_relevance_score(chunk), section_id, chunk))

    if not ranked:
        # Fallback: strict filter yielded nothing — use all chunks ranked by score.
        ranked = [
            (0, 0, -_chunk_relevance_score(chunk), str(chunk.get("section_id", "")), chunk)
            for chunk in chunks
        ]

    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [item[4] for item in ranked[:top_k]]


def _sara_max_tokens(case: EvalCase) -> int:
    """Return an appropriate max-token cap for a SARA case.

    SARA label cases need only a short label + brief reasoning; numeric cases
    need arithmetic steps. Capping output avoids wasting GPU time on padding.
    """
    expected_type = str(case.metadata.get("expected_type", "freeform")).lower()
    return {
        "label":   cfg.SARA_MAX_TOKENS_LABEL,
        "numeric": cfg.SARA_MAX_TOKENS_NUMERIC,
        "string":  cfg.SARA_MAX_TOKENS_STRING,
    }.get(expected_type, cfg.SARA_MAX_TOKENS_DEFAULT)


def run_case(
    case: EvalCase,
    dataset,
    retriever: HybridRetriever | None,
    model_id: str,
    mode: str,
    dry_run: bool,
    judge_model_id: str,
    skip_scoring: bool,
) -> dict:
    """Run one case end-to-end and return a serializable result dict."""
    is_sara = dataset.name == "sara_v3"

    retrieval_query = case.question
    if is_sara and cfg.SARA_APPEND_TEXT_CONTEXT_TO_RETRIEVAL:
        facts = str(case.metadata.get("context", "")).strip()
        if facts:
            retrieval_query = f"{case.question}\n{facts}"

    # Retrieve context (mode=none returns empty chunks without touching the index)
    if mode == "none" or retriever is None:
        chunks = []
    else:
        chunks = retriever.query(
            retrieval_query,
            k=cfg.TOP_K_VECTOR,
            depth=cfg.BFS_DEPTH,
            mode=mode,
        )
        chunks = _dedupe_chunks_by_section(chunks)
        if is_sara:
            chunks = _prioritize_sara_supplement_chunks(chunks, case, mode=mode)
        else:
            chunks = chunks[: cfg.TOP_K_VECTOR]
    retrieved_ids = [c["section_id"] for c in chunks]

    # Generate response
    if dry_run:
        response = _MOCK_RESPONSE
    else:
        system = build_system_prompt(chunks)
        user_prompt = _build_sara_user_prompt(case) if is_sara else case.question
        if is_sara:
            system = f"{system}{_build_sara_system_append(case)}"
        max_tokens = _sara_max_tokens(case) if is_sara else None
        response = call_llm(system, user_prompt, model_id, max_tokens=max_tokens)

    # Score via the dataset's own score() method.
    # TaxBench returns both LLM-as-judge fields AND rouge1/rouge2/rougeL.
    # Other datasets may override score() for exact-match or numeric scoring.
    if skip_scoring or dry_run:
        scoring = {"earned": None, "total": None, "feedback": "Scoring skipped."}
    else:
        judge_fn = lambda resp, rubric: judge_response(resp, rubric, judge_model_id)
        scoring = dataset.score(response, case, judge_fn)

    # Retrieval metrics — only when the case has ground-truth relevant IDs.
    retrieved_sources = sorted({chunk.get("source", "") for chunk in chunks if chunk.get("source")})
    retrieval_metrics = _compute_retrieval_metrics(
        retrieved_ids,
        case.relevant_ids,
        retrieved_sources,
    )
    citation_metrics = _compute_citation_metrics(response, retrieved_ids)

    return {
        "id":                case.id,
        "question":          case.question,
        "rubric":            case.rubric,
        "response":          response,
        "mode":              mode,
        "model":             model_id,
        "retrieved_ids":     retrieved_ids,
        "n_chunks":          len(chunks),
        "retrieval_metrics": retrieval_metrics,
        "citation_metrics":  citation_metrics,
        "scoring":           scoring,
    }


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]) -> None:
    """Print per-mode summary statistics after a run completes."""
    if not results:
        print("  No results.")
        return

    # Judge score — exclude parse failures from the average and report them.
    parse_errors = sum(1 for r in results if r["scoring"].get("_parse_error"))
    if parse_errors:
        print(f"  Judge parse errors : {parse_errors} case(s) excluded from score average")

    scored = [r for r in results if r["scoring"].get("earned") is not None]
    if scored:
        total_earned = sum(r["scoring"]["earned"] for r in scored)
        total_max    = sum(r["scoring"]["total"]  for r in scored)
        pct = (total_earned / total_max * 100) if total_max > 0 else 0.0
        print(f"  Judge score : {total_earned:.2f} / {total_max:.2f}  ({pct:.1f}%)"
              f"  [{len(scored)}/{len(results)} cases]")

    # ROUGE — reported only if present (TaxBench and future datasets that add it)
    rouge_results = [r for r in results if "rouge1" in r["scoring"]]
    if rouge_results:
        for metric in ("rouge1", "rouge2", "rougeL"):
            avg = sum(r["scoring"][metric] for r in rouge_results) / len(rouge_results)
            print(f"  {metric:<8}: {avg:.4f}  (avg over {len(rouge_results)} cases)")

    # Retrieval metrics — recall@k and MRR (requires ground-truth relevant_ids)
    retrieval_cases = [r for r in results if r.get("retrieval_metrics")]
    if retrieval_cases:
        for metric in ("recall_at_k", "mrr"):
            avg = sum(r["retrieval_metrics"][metric] for r in retrieval_cases) / len(retrieval_cases)
            label = "recall@k(hier)" if metric == "recall_at_k" else "MRR(hier)   "
            print(f"  {label}: {avg:.4f}  (avg over {len(retrieval_cases)} cases)")

        exact_cases = [
            r
            for r in retrieval_cases
            if "recall_at_k_exact" in r["retrieval_metrics"] and "mrr_exact" in r["retrieval_metrics"]
        ]
        if exact_cases:
            avg_recall_exact = (
                sum(r["retrieval_metrics"]["recall_at_k_exact"] for r in exact_cases)
                / len(exact_cases)
            )
            avg_mrr_exact = (
                sum(r["retrieval_metrics"]["mrr_exact"] for r in exact_cases)
                / len(exact_cases)
            )
            print(
                f"  recall@k(exact): {avg_recall_exact:.4f}  "
                f"(avg over {len(exact_cases)} cases)"
            )
            print(
                f"  MRR(exact)   : {avg_mrr_exact:.4f}  "
                f"(avg over {len(exact_cases)} cases)"
            )

        source_recall_cases = [
            r for r in retrieval_cases if "source_recall" in r["retrieval_metrics"]
        ]
        if source_recall_cases:
            avg_source_recall = (
                sum(r["retrieval_metrics"]["source_recall"] for r in source_recall_cases)
                / len(source_recall_cases)
            )
            print(
                f"  source_recall: {avg_source_recall:.4f}  "
                f"(avg over {len(source_recall_cases)} cases)"
            )

            per_source_scores: dict[str, list[float]] = defaultdict(list)
            for result in source_recall_cases:
                for source, score in result["retrieval_metrics"].get("source_hits", {}).items():
                    per_source_scores[source].append(float(score))

            if per_source_scores:
                pretty = ", ".join(
                    f"{source}:{(sum(scores)/len(scores)):.2f}"
                    for source, scores in sorted(per_source_scores.items())
                )
                print(f"  source_recall_by_source: {pretty}")

    citation_cases = [r for r in results if r.get("citation_metrics")]
    if citation_cases:
        precision_values = [
            r["citation_metrics"]["citation_precision"]
            for r in citation_cases
            if r["citation_metrics"].get("citation_precision") is not None
        ]
        if precision_values:
            avg_precision = sum(precision_values) / len(precision_values)
            print(
                f"  citation_precision: {avg_precision:.4f}  "
                f"(avg over {len(precision_values)} cited cases)"
            )

        grounding_values = [
            r["citation_metrics"]["grounding_rate"]
            for r in citation_cases
            if r["citation_metrics"].get("grounding_rate") is not None
        ]
        if grounding_values:
            avg_grounding = sum(grounding_values) / len(grounding_values)
            print(
                f"  grounding_rate: {avg_grounding:.4f}  "
                f"(avg over {len(grounding_values)} citation-bearing cases)"
            )

    for key, label in (
        ("score_before_citation_penalty", "score_before_citation_penalty"),
        ("score_after_citation_penalty", "score_after_citation_penalty"),
        ("citation_penalty", "citation_penalty"),
        ("answer_correct", "answer_correct"),
        ("calculation_steps_present", "calc_steps"),
        ("citation_fact_precision", "citation_fact_precision"),
        ("citation_fact_recall", "citation_fact_recall"),
    ):
        values = [
            float(r["scoring"][key])
            for r in results
            if key in r.get("scoring", {}) and r["scoring"].get(key) is not None
        ]
        if values:
            avg = sum(values) / len(values)
            print(f"  {label}: {avg:.4f}  (avg over {len(values)} cases)")

    predicted_labels = [
        str(r["scoring"].get("predicted_label", "")).lower()
        for r in results
        if r.get("scoring", {}).get("predicted_label") is not None
    ]
    if predicted_labels:
        unknown_rate = predicted_labels.count("unknown") / len(predicted_labels)
        print(
            f"  unknown_label_rate: {unknown_rate:.4f}  "
            f"(avg over {len(predicted_labels)} label cases)"
        )


def _collect_mode_summary(results: list[dict]) -> dict:
    """Collect compact per-mode metrics for cross-mode comparison."""
    summary: dict[str, float] = {}

    scored = [r for r in results if r.get("scoring", {}).get("earned") is not None]
    if scored:
        earned = sum(float(r["scoring"]["earned"]) for r in scored)
        total = sum(float(r["scoring"].get("total") or 0.0) for r in scored)
        if total > 0:
            summary["score_pct"] = (earned / total) * 100.0

    for key in (
        "score_before_citation_penalty",
        "score_after_citation_penalty",
        "citation_penalty",
        "answer_correct",
        "calculation_steps_present",
        "citation_fact_precision",
        "citation_fact_recall",
        "unknown_label_rate",
        "recall_at_k",
        "mrr",
        "recall_at_k_exact",
        "mrr_exact",
    ):
        if key in {"recall_at_k", "mrr", "recall_at_k_exact", "mrr_exact"}:
            values = [
                float(r["retrieval_metrics"][key])
                for r in results
                if key in r.get("retrieval_metrics", {})
            ]
        elif key == "unknown_label_rate":
            labels = [
                str(r["scoring"].get("predicted_label", "")).lower()
                for r in results
                if r.get("scoring", {}).get("predicted_label") is not None
            ]
            values = [labels.count("unknown") / len(labels)] if labels else []
        else:
            values = [
                float(r["scoring"][key])
                for r in results
                if key in r.get("scoring", {}) and r["scoring"].get(key) is not None
            ]

        if values:
            summary[key] = sum(values) / len(values)

    return summary


def _print_mode_comparison(mode_summaries: dict[str, dict]) -> None:
    """Print concise mode comparison and hybrid-vs-none deltas."""
    if not mode_summaries:
        return

    print("\n=== Mode Comparison ===")
    metrics_order = [
        "score_pct",
        "score_before_citation_penalty",
        "score_after_citation_penalty",
        "citation_penalty",
        "answer_correct",
        "unknown_label_rate",
        "calculation_steps_present",
        "citation_fact_precision",
        "citation_fact_recall",
        "recall_at_k",
        "mrr",
        "recall_at_k_exact",
        "mrr_exact",
    ]

    for mode in ALL_MODES:
        if mode not in mode_summaries:
            continue
        parts = []
        summary = mode_summaries[mode]
        for metric in metrics_order:
            if metric in summary:
                parts.append(f"{metric}={summary[metric]:.4f}")
        if parts:
            print(f"  {mode:<6}: " + ", ".join(parts))

    if "none" in mode_summaries and "hybrid" in mode_summaries:
        print("\n  Hybrid improvement vs none:")
        base = mode_summaries["none"]
        hybrid = mode_summaries["hybrid"]
        for metric in metrics_order:
            if metric in base and metric in hybrid:
                delta = hybrid[metric] - base[metric]
                sign = "+" if delta >= 0 else ""
                print(f"    {metric}: {sign}{delta:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation for the tax GraphRAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=list(REGISTRY),
        required=True,
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--mode",
        choices=ALL_MODES + ["all"],
        default="hybrid",
        help="Retrieval mode. 'all' runs all four modes back-to-back.",
    )
    parser.add_argument(
        "--model",
        default="claude",
        help=(
            "LLM for response generation. "
            "Accepted: claude | gemini | ollama | ollama:<model-name>"
        ),
    )
    parser.add_argument(
        "--judge",
        default="claude",
        help=(
            "LLM for rubric scoring. "
            "Accepted: claude | gemini | ollama | ollama:<model-name>"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N cases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls; write mock responses to test the pipeline.",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Generate responses but skip the LLM-as-judge scoring step.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing result file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "dataset",
        help="Path to the dataset directory (default: ../dataset/ from repo root).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory to write result JSON files.",
    )
    return parser.parse_args()


def _resolve_model_cli(value: str, default_ollama: str) -> str:
    """Resolve CLI model value to an internal model id string."""
    raw = str(value or "").strip()
    lowered = raw.lower()

    if lowered == "claude":
        return cfg.CLAUDE_MODEL
    if lowered == "gemini":
        return cfg.GEMINI_MODEL
    if lowered == "ollama":
        return f"ollama:{default_ollama}"
    if lowered.startswith("ollama:") and len(raw) > len("ollama:"):
        return raw

    raise ValueError(
        "Invalid model value. Use one of: claude, gemini, ollama, ollama:<model-name>."
    )


def main() -> None:
    args = parse_args()

    model_id = _resolve_model_cli(args.model, cfg.OLLAMA_MODEL)
    judge_id = _resolve_model_cli(args.judge, cfg.OLLAMA_MODEL)

    modes     = ALL_MODES         if args.mode  == "all"     else [args.mode]

    # Load dataset and inject judge model so adapters can call LLM directly.
    dataset_cls: type = REGISTRY[args.dataset]
    dataset: Dataset  = dataset_cls()
    dataset.judge_model_id = judge_id
    cases = dataset.load(args.data_dir, limit=args.limit)
    print(f"Loaded {len(cases)} cases from '{dataset.name}'")

    # Load retriever only if at least one mode needs it (mode=none skips the index)
    needs_retriever = any(m != "none" for m in modes)
    if needs_retriever:
        print("Loading retriever...", end=" ", flush=True)
        retriever = HybridRetriever.load(cfg)
        print("ready.")
    else:
        retriever = None
        print("Retriever not needed for mode=none; skipping index load.")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    mode_summaries: dict[str, dict] = {}
    _print_lock = threading.Lock()
    # Map case_id → 1-based position in original cases list (for progress display)
    _case_position = {c.id: i for i, c in enumerate(cases, 1)}

    for mode in modes:
        tag = "dryrun" if args.dry_run else args.model
        out_path = args.results_dir / f"{dataset.name}__{tag}__{mode}.json"
        partial_path = out_path.with_suffix(".partial.jsonl")

        # --- Skip / resume / overwrite logic ---
        if out_path.exists() and not partial_path.exists() and not args.overwrite:
            print(f"Skipping {out_path.name} (already complete; use --overwrite to redo)")
            continue

        completed_results: dict[str, dict] = {}
        if partial_path.exists() and not args.overwrite:
            for line in partial_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "id" in entry:
                        completed_results[entry["id"]] = entry
                except (json.JSONDecodeError, KeyError):
                    pass
            if completed_results:
                print(f"  Resuming: {len(completed_results)}/{len(cases)} cases already done.")
        elif args.overwrite:
            partial_path.unlink(missing_ok=True)
            if out_path.exists():
                out_path.unlink()

        pending = [c for c in cases if c.id not in completed_results]
        n = len(cases)

        print(f"\n--- {dataset.name} | model={model_id} | mode={mode} "
              f"| workers={cfg.EVAL_CONCURRENCY} | todo={len(pending)}/{n} ---")

        _progress_lock = threading.Lock()

        def _append_progress(result: dict, _path: Path = partial_path) -> None:
            with _progress_lock:
                with _path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(result) + "\n")

        def _run_one(case: EvalCase, _mode: str = mode) -> tuple[str, dict]:
            """Run one case and return (case_id, result). Thread-safe."""
            idx = _case_position[case.id]
            t0 = time.monotonic()
            with _print_lock:
                print(f"  [{idx}/{n}] {case.id[:50]}", flush=True)
            try:
                result = run_case(
                    case=case,
                    dataset=dataset,
                    retriever=retriever,
                    model_id=model_id,
                    mode=_mode,
                    dry_run=args.dry_run,
                    judge_model_id=judge_id,
                    skip_scoring=args.skip_scoring,
                )
            except Exception as exc:
                with _print_lock:
                    print(f"    [{idx}] ERROR: {exc}", flush=True)
                result = {
                    "id": case.id,
                    "question": case.question,
                    "rubric": case.rubric,
                    "response": "",
                    "mode": _mode,
                    "model": model_id,
                    "retrieved_ids": [],
                    "n_chunks": 0,
                    "retrieval_metrics": {},
                    "citation_metrics": {},
                    "scoring": {
                        "earned": None,
                        "total": None,
                        "feedback": f"Case failed: {exc}",
                        "_case_error": True,
                    },
                    "error": str(exc),
                }
            elapsed = time.monotonic() - t0
            with _print_lock:
                print(f"    [{idx}] done in {elapsed:.1f}s", flush=True)
            _append_progress(result)
            return case.id, result

        new_results: dict[str, dict] = {}
        if pending:
            if cfg.EVAL_CONCURRENCY <= 1:
                for case in pending:
                    case_id, result = _run_one(case)
                    new_results[case_id] = result
            else:
                with ThreadPoolExecutor(max_workers=cfg.EVAL_CONCURRENCY) as pool:
                    futures = {pool.submit(_run_one, case): case.id for case in pending}
                    for future in as_completed(futures):
                        case_id, result = future.result()
                        new_results[case_id] = result

        # Merge completed + new, preserving original case order
        all_results = {**completed_results, **new_results}
        results = [all_results[c.id] for c in cases if c.id in all_results]

        _print_summary(results)
        mode_summaries[mode] = _collect_mode_summary(results)

        payload = {
            "dataset": dataset.name,
            "model":   model_id,
            "mode":    mode,
            "n_cases": len(results),
            "cases":   results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        partial_path.unlink(missing_ok=True)
        print(f"  Saved -> {out_path}")

    if len(mode_summaries) > 1:
        _print_mode_comparison(mode_summaries)

    print("\nDone.")


if __name__ == "__main__":
    main()
