"""IRS form instruction Q&A dataset adapter.

Loads test-tax_form_instructions_qa_pairs.parquet (200 cases).
Each row has a passage drawn from a 2024 IRS form instruction, a question
about that passage, and a reference answer.

Schema:
  source    - document title, e.g. "2024 Instructions for Schedule C (2024)"
  filename  - source file, e.g. "i1040sc.txt"
  context   - the IRS passage the Q&A was derived from
  question  - question about the passage
  answer    - reference answer (clean string, no [+N.NN] markers)

Scoring:
  LLM-as-judge  - compares model response to the reference answer on a 0-1
                  scale; prompt is simpler than TaxBench (no rubric criteria).
  ROUGE (1/2/L) - lexical overlap between model response and reference answer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

from evaluation.datasets.base import Dataset, EvalCase


# ---------------------------------------------------------------------------
# Judge prompt (reference-answer style, not rubric style)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are evaluating a tax Q&A response against a reference answer.

Question:
{question}

Reference answer:
{reference}

Model response:
{response}

Score the model response on a scale of 0.0 to 1.0:
  1.0  - fully correct and consistent with the reference answer
  0.5  - partially correct; captures the main point but misses details
  0.0  - incorrect or contradicts the reference answer

Return a JSON object with exactly these fields:
  "score"    - float in [0.0, 1.0]
  "feedback" - one sentence explaining the score

Return only valid JSON, no extra text."""


def _judge_with_reference(
    response: str, question: str, reference: str, model_id: str
) -> dict:
    """Call Claude or Gemini to score a response against a reference answer."""
    prompt = _JUDGE_PROMPT.format(
        question=question, reference=reference, response=response
    )
    if model_id.startswith("claude"):
        raw = _call_claude(prompt, model_id)
    elif model_id.startswith("gemini"):
        raw = _call_gemini(prompt, model_id)
    elif model_id.startswith("ollama:"):
        raw = _call_ollama(prompt, model_id)
    else:
        return {"earned": None, "total": 1.0, "feedback": f"Unknown model: {model_id}"}

    return _parse_judge_json(raw)


def _call_claude(prompt: str, model_id: str) -> str:
    from src import config as cfg
    import anthropic
    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=model_id,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _call_gemini(prompt: str, model_id: str) -> str:
    from src import config as cfg
    from google import genai
    client = genai.Client(api_key=cfg.GEMINI_API_KEY)
    resp = client.models.generate_content(model=model_id, contents=prompt)
    return resp.text.strip()


def _call_ollama(prompt: str, model_id: str) -> str:
    from src import config as cfg

    model_name = model_id.split("ollama:", 1)[1] if model_id.startswith("ollama:") else model_id
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Return only valid JSON matching the requested schema."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": cfg.OLLAMA_THINK,
    }

    body = json.dumps(payload).encode("utf-8")
    endpoint = cfg.OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
    req = urlrequest.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=cfg.OLLAMA_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
    except urlerror.URLError as exc:
        raise RuntimeError(
            "Failed to reach Ollama server. "
            f"Check OLLAMA_BASE_URL ({cfg.OLLAMA_BASE_URL}) and run 'ollama serve'."
        ) from exc

    data = json.loads(raw)
    return str(data.get("message", {}).get("content", "")).strip()


def _parse_judge_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        result = json.loads(raw)
        score = float(result.get("score", 0.0))
        return {
            "earned":   round(score, 4),
            "total":    1.0,
            "feedback": str(result.get("feedback", "")),
        }
    except (json.JSONDecodeError, ValueError, KeyError):
        return {
            "earned":   0.0,
            "total":    1.0,
            "feedback": f"Judge response could not be parsed: {raw[:200]}",
        }


# ---------------------------------------------------------------------------
# ROUGE scoring (reused from taxbench pattern)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------

class IRSFormQADataset(Dataset):
    """Adapter for test-tax_form_instructions_qa_pairs.parquet."""

    name = "irs_form_qa"

    def load(self, data_dir: Path, limit: int | None = None) -> list[EvalCase]:
        path = data_dir / "test-tax_form_instructions_qa_pairs.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"IRS QA dataset not found at {path}. "
                "Expected test-tax_form_instructions_qa_pairs.parquet "
                "inside the dataset/ directory."
            )

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas and pyarrow are required to load the parquet file. "
                "Run: pip install pandas pyarrow"
            ) from exc

        df = pd.read_parquet(path)
        if limit is not None:
            df = df.head(limit)

        cases: list[EvalCase] = []
        for i, row in df.iterrows():
            cases.append(
                EvalCase(
                    id=f"irs_qa_{i}",
                    question=str(row["question"]),
                    # rubric stores the reference answer for ROUGE + judge
                    rubric=str(row["answer"]),
                    metadata={
                        "source":   str(row.get("source", "")),
                        "filename": str(row.get("filename", "")),
                        "context":  str(row.get("context", "")),
                    },
                )
            )

        return cases

    def score(self, response: str, case: EvalCase, judge_fn) -> dict:
        """Score using reference-answer LLM judge + ROUGE.

        Uses self.judge_model_id (injected by run_eval.py) to call the LLM
        with a reference-answer prompt rather than the rubric-criteria prompt
        used by TaxBench.
        """
        # LLM-as-judge with reference-answer prompt
        if case.rubric and self.judge_model_id:
            judge_result = _judge_with_reference(
                response=response,
                question=case.question,
                reference=case.rubric,
                model_id=self.judge_model_id,
            )
        else:
            judge_result = {"earned": None, "total": 1.0, "feedback": "Scoring skipped."}

        # ROUGE vs reference answer
        rouge_result: dict = {}
        if case.rubric:
            rouge_result = _compute_rouge(response, case.rubric)

        return {**judge_result, **rouge_result}
