# GraphRAG Tax Assistant

Hybrid GraphRAG system for federal income tax Q&A.  The system retrieves
applicable IRC sections and IRS publication guidance for a taxpayer's situation,
then grounds an LLM's answer in those specific sources.

**Scope:** individual resident filers, Form 1040. Data can be loaded from
knowledge profiles (for example `knowledge/2017` or `knowledge/2024-2026`).

See [architecture.md](architecture.md) for full pipeline diagrams.

---

## How it works

```text
IRS XML/PDF publications + Title 26 USC XML
          |
          | parse   →  usc26_parser.py, irs_xml_parser.py, irs_pdf_parser.py
          | normalize → normalizer.py (cross-ref extraction)
          | chunk   →  chunker.py (split at sentence boundaries)
          v
     annotated chunks
          |
     +----|-----+
     |          |
     v          v
  FAISS         NetworkX
  vector index  graph index
  (content +    (hierarchy,
  section_id)   xref,
                coverage edges)
     |          |
     +----+-----+
          |
          | hybrid_retriever.py (merge + re-rank)
          v
     top-k relevant chunks
          |
          | prepended as system context
          v
     LLM (Claude / Gemini / Ollama)
          |
          v
     answer with specific IRC and IRS citations
```

---

## Step-by-step setup

### 1. Create and activate the conda environment

```bash
conda create -n cs789_research python=3.11
conda activate cs789_research
```

### 2. Install Python dependencies

Run from inside `graphrag_tax/`:

```bash
cd graphrag_tax
pip install -r requirements.txt
```

### 3. Download the spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 4. Set up API keys (optional for Ollama)

Create a `.env` file at the **repository root** (`research_789/`), not inside `graphrag_tax/`:

```text
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AI...
```

### 4b. (Optional) Set up local open-source model with Ollama

```bash
# One-time model download
ollama pull qwen2.5:3b

# Start server if not already running
ollama serve
```

### 5. Build the knowledge graph and vector index

This is a one-time step.  It parses available XML/PDF sources, builds the FAISS vector
index, and constructs the NetworkX knowledge graph.

By default, artifacts are profile-scoped under `data/processed/<profile>/`
(for example `data/processed/2017/`). Set `PROFILED_OUTPUTS=0` to write to
`data/processed/` directly.

```bash
python scripts/build_pipeline.py
```

To run a specific knowledge profile:

```bash
# 2017 profile (USC XML + IRS PDFs in knowledge/2017)
KNOWLEDGE_PROFILE=2017 python scripts/build_pipeline.py

# 2024-2026 profile (USC XML + IRS XML sources in knowledge/2024-2026)
KNOWLEDGE_PROFILE=2024-2026 python scripts/build_pipeline.py

# Optional: disable profile-scoped output directories
PROFILED_OUTPUTS=0 KNOWLEDGE_PROFILE=2017 python scripts/build_pipeline.py
```

Expected output:

```text
Parsing Title 26 XML (Subtitle A)...
  4831 statute chunks
Discovering IRS XML publications in knowledge/...
  Found 36 IRS sources: [p17, p501, p502, ...]
  ...
Embedding 14129 chunks (content)...
  Outer chunks (content): 100%|...
Building GraphRAG knowledge graph with community detection...
  Section linker added N coverage + M cross-pub edges
Graph audit passed: 37 sources connected
Pipeline complete. Artifacts saved to: data/processed/
```

Expected runtime: 10–20 minutes on first run (embedding generation).

> **Intel Mac / Windows note:** The pipeline auto-detects Intel Mac and switches
> to `all-MiniLM-L6-v2` (smaller, faster) and reduces batch sizes.
> To force low-resource mode manually: `LOW_RESOURCE_MODE=1 python scripts/build_pipeline.py`

---

## Chatbot

Interactive multi-turn assistant. Requires the build step and either API keys
(Claude/Gemini) or a local Ollama model.

```bash
python chatbot.py
```

Options:

```bash
python chatbot.py --mode hybrid    # default: graph + vector retrieval
python chatbot.py --mode vector    # vector index only
python chatbot.py --mode graph     # graph traversal only
python chatbot.py --mode none      # LLM only, no retrieval (baseline)
python chatbot.py --compare        # side-by-side: hybrid vs. LLM-only
python chatbot.py --model gemini   # use Gemini instead of Claude
python chatbot.py --model ollama   # use local model from OLLAMA_MODEL
python chatbot.py --top-k 15       # retrieve more chunks per query

# choose a different local model
OLLAMA_MODEL=llama3.2:3b python chatbot.py --model ollama
```

> Local Ollama inference is usually free per token (no provider billing), but
> runtime is still limited by your machine's CPU/GPU, RAM/VRAM, and model size.

In-session commands:

| Command                               | Action                       |
| ------------------------------------- | ---------------------------- |
| `/mode <none\|vector\|graph\|hybrid>` | Switch retrieval mode        |
| `/sources`                            | Show sources from last query |
| `/clear`                              | Reset conversation history   |
| `/quit`                               | Exit                         |

---

## Smoke test (no API key needed)

Test that the retrieval pipeline loaded correctly before spending API credits:

```bash
python scripts/test_query.py "standard deduction for single filers 2024"
python scripts/test_query.py "earned income credit requirements" --mode graph --k 8
python scripts/test_query.py "§32 EIC phase-out" --mode hybrid
```

Visualize the knowledge graph in a browser:

```bash
python scripts/viz_graph.py --sample-n 300
# opens data/processed/graph_view.html
```

---

## Evaluation

Batch evaluation compares the four retrieval modes (none / vector / graph /
hybrid) to measure GraphRAG's contribution.  Results are saved to
`evaluation/results/`.

### Available datasets

| `--dataset` flag | Source file | Cases | Scoring |
| ---------------- | ----------- | ----- | ------- |
| `taxbench` | `dataset/TaxBench-EvalSet.jsonl` | varies | LLM-as-judge (rubric criteria) + ROUGE-1/2/L + recall@k + MRR |
| `irs_form_qa` | `dataset/test-tax_form_instructions_qa_pairs.parquet` | 200 | LLM-as-judge (reference answer, 0–1 scale) + ROUGE-1/2/L |
| `sara_v3` | `dataset/sara_v3/` | split-dependent | Deterministic scoring for numeric/entailment cases + ROUGE-1/2/L + recall@k + MRR + citation metrics |

### Run TaxBench evaluation

```bash
# Hybrid mode, Claude
python evaluation/run_eval.py --dataset taxbench --mode hybrid --model claude

# All four modes back-to-back
python evaluation/run_eval.py --dataset taxbench --mode all --model claude

# Dry run to test the pipeline without spending API credits
python evaluation/run_eval.py --dataset taxbench --mode hybrid --dry-run

# First 10 cases only
python evaluation/run_eval.py --dataset taxbench --mode hybrid --limit 10

# Skip the LLM-as-judge scoring step (generate responses only)
python evaluation/run_eval.py --dataset taxbench --mode all --skip-scoring

# Local open-source generation + local judge
python evaluation/run_eval.py --dataset taxbench --mode hybrid --model ollama --judge ollama --limit 5
```

### Run IRS Form Q&A evaluation

```bash
# Hybrid mode, Claude judge
python evaluation/run_eval.py --dataset irs_form_qa --mode hybrid --model claude

# All four modes back-to-back
python evaluation/run_eval.py --dataset irs_form_qa --mode all --model claude

# Dry run — no index needed, no API credits spent
python evaluation/run_eval.py --dataset irs_form_qa --mode none --dry-run --limit 5
```

> **Note:** `--mode none` never loads the FAISS index, so dry runs work even
> before the build step has been run.

### Run SARA v3 evaluation

```bash
# Default split is test
python evaluation/run_eval.py --dataset sara_v3 --mode none --dry-run --limit 20

# Choose split via environment variable
SARA_SPLIT=train python evaluation/run_eval.py --dataset sara_v3 --mode none --dry-run --limit 20

# Use retrieval modes once indexes are available
SARA_SPLIT=test python evaluation/run_eval.py --dataset sara_v3 --mode hybrid --model gemini --limit 5

# Same run with local open-source model
SARA_SPLIT=test python evaluation/run_eval.py --dataset sara_v3 --mode hybrid --model ollama --judge ollama --limit 5
```

Result files are written to:

```text
evaluation/results/taxbench__claude__hybrid.json
evaluation/results/taxbench__claude__none.json
evaluation/results/irs_form_qa__claude__hybrid.json
...
```

### Adding a new dataset

1. Create `evaluation/datasets/<name>.py` implementing the `Dataset` interface
   (see [evaluation/datasets/base.py](evaluation/datasets/base.py)).
2. Import the class and add it to `REGISTRY` in
   [evaluation/datasets/\_\_init\_\_.py](evaluation/datasets/__init__.py).
3. Pass `--dataset <name>` to `run_eval.py`.

To use a custom judge prompt (as `irs_form_qa` does), override `score()` and
call the LLM directly using `self.judge_model_id` (injected at runtime).

To get recall@k and MRR for your dataset, populate `EvalCase.relevant_ids`
with known ground-truth section IDs (format: `"26 usc §32"`). TaxBench
extracts these automatically from its rubric text.

---

## Workflows

The project includes a `Makefile` for common tasks:

```bash
make install          # install dependencies
make spacy            # download spaCy model
make build-2017       # build pipeline with knowledge/2017
make build-2024       # build pipeline with knowledge/2024-2026
make smoke            # quick retrieval smoke test
make test             # run pytest suite
make eval-sara-dry    # run SARA evaluation dry-run (mode none)
make eval-taxbench-dry
```

---

## Project structure

```text
research_789/                     repository root
├── graphrag_tax/                 this project
│   ├── src/
│   │   ├── config.py             all settings, file paths, and tuning constants
│   │   ├── utils/
│   │   │   └── ref_patterns.py   shared USC/IRS regex (used by normalizer + retriever)
│   │   ├── ingestion/            parse IRS and USC sources into raw chunks
│   │   │   ├── usc26_parser.py
│   │   │   ├── irs_xml_parser.py
│   │   │   └── irs_pdf_parser.py
│   │   ├── preprocessing/        extract cross-refs, split long chunks
│   │   │   ├── normalizer.py
│   │   │   └── chunker.py
│   │   ├── indexing/             build indexes at pipeline time
│   │   │   ├── vector_index.py   FAISS dual-embedding index
│   │   │   ├── graph_index.py    NetworkX DiGraph + Louvain communities
│   │   │   ├── section_linker.py inject coverage + cross-pub edges
│   │   │   └── graph_audit.py    build-time connectivity validation
│   │   └── retrieval/            query-time retrieval
│   │       ├── vector_retriever.py
│   │       ├── graph_retriever.py
│   │       └── hybrid_retriever.py
│   ├── scripts/
│   │   ├── build_pipeline.py     one-time index build
│   │   ├── test_query.py         smoke-test retrieval
│   │   └── viz_graph.py          write graph_view.html
│   ├── evaluation/
│   │   ├── run_eval.py           batch evaluation runner
│   │   ├── datasets/
│   │   │   ├── base.py           Dataset abstract class + EvalCase (relevant_ids)
│   │   │   ├── taxbench.py       TaxBench adapter (rubric judge + ROUGE + recall@k/MRR)
│   │   │   ├── irs_form_qa.py    IRS Form QA adapter (reference judge + ROUGE)
│   │   │   ├── sara_v3.py        SARA adapter (split files + deterministic scoring)
│   │   │   └── __init__.py       dataset registry
│   │   └── results/              output JSONs (gitignored)
│   ├── data/processed/           pipeline artifacts (gitignored)
│   ├── chatbot.py                interactive CLI
│   ├── requirements.txt
│   ├── README.md                 this file
│   └── architecture.md           pipeline diagrams
├── knowledge/                    IRS XML + USC XML (shared source data)
├── data_syn/                     synthetic benchmark generation
├── tax-calc-bench/               TaxCalcBench evaluation harness
└── dataset/                      evaluation datasets
```

---

## Configuration

All settings live in [src/config.py](src/config.py).

| Setting                        | Default             | Notes                                       |
| ------------------------------ | ------------------- | ------------------------------------------- |
| `MAX_CHUNK_CHARS`              | 2000                | Chunk size cap                              |
| `EMBEDDING_MODEL`              | `all-mpnet-base-v2` | `all-MiniLM-L6-v2` on Intel Mac             |
| `TOP_K_VECTOR`                 | 10                  | Chunks returned per query                   |
| `BFS_DEPTH`                    | 2                   | Graph traversal hops                        |
| `EXCLUDED_SOURCES`             | `{i1040nr, p519}`   | Non-resident sources, excluded              |
| `LOW_RESOURCE_MODE`            | auto (Intel Mac)    | Smaller model + smaller batches             |
| `OLLAMA_MODEL`                 | `qwen2.5:3b`        | Local model used when `--model ollama`      |
| `HYBRID_ALPHA_DEFAULT`         | 0.6                 | Vector weight for broad/semantic queries    |
| `HYBRID_ALPHA_SECTION_REF`     | 0.35                | Vector weight when query cites explicit §   |
| `GRAPH_EDGE_WEIGHT_XREF`       | 3.0                 | BFS neighbor priority for xref edges        |
| `GRAPH_EDGE_WEIGHT_COVERAGE`   | 2.5                 | BFS neighbor priority for coverage edges    |
| `GRAPH_EDGE_WEIGHT_HIERARCHY`  | 1.8                 | BFS neighbor priority for hierarchy edges   |
| `GRAPH_OVERLAP_WEIGHT`         | 0.20                | Per-term overlap contribution to BFS score  |
| `GRAPH_USC_BOOST`              | 0.15                | Authority boost for USC26 nodes             |

To add a new IRS XML source: place its XML at `knowledge/<name>/<name>.xml`
and add a display label to `src/ingestion/irs_xml_parser.SOURCE_LABELS`.

For local inference endpoint configuration, set `OLLAMA_BASE_URL`
(default: `http://localhost:11434`).
