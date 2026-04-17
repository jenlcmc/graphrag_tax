from pathlib import Path
from dotenv import load_dotenv
import os
import platform

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
	raw = os.getenv(name)
	if raw is None:
		return default
	return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
	raw = os.getenv(name)
	if raw is None:
		return default
	try:
		return int(raw)
	except ValueError:
		return default


def _env_float(name: str, default: float) -> float:
	raw = os.getenv(name)
	if raw is None:
		return default
	try:
		return float(raw)
	except ValueError:
		return default


def _slugify(value: str) -> str:
	value = value.strip().lower()
	value = value.replace(" ", "-")
	value = "".join(ch for ch in value if ch.isalnum() or ch in {"-", "_"})
	return value or "default"


def _resolve_knowledge_dir(root: Path) -> Path:
	"""Resolve the active knowledge directory.

	Selection order:
	  1) KNOWLEDGE_PROFILE environment variable (e.g. "2017", "2024-2026")
	  2) Legacy flat layout: knowledge/usc26.xml
	  3) Preferred profiles when present: knowledge/2017, knowledge/2024-2026
	  4) First child directory containing usc26.xml
	"""
	knowledge_root = root.parent / "knowledge"
	profile = os.getenv("KNOWLEDGE_PROFILE", "").strip()

	if profile:
		candidate = knowledge_root / profile
		if not candidate.exists():
			raise FileNotFoundError(
				f"KNOWLEDGE_PROFILE '{profile}' not found under {knowledge_root}"
			)
		return candidate

	if (knowledge_root / "usc26.xml").exists():
		return knowledge_root

	for preferred in ("2017", "2024-2026"):
		candidate = knowledge_root / preferred
		if (candidate / "usc26.xml").exists():
			return candidate

	for child in sorted(knowledge_root.iterdir()):
		if child.is_dir() and (child / "usc26.xml").exists():
			return child

	return knowledge_root

ROOT          = Path(__file__).parent.parent
# knowledge/ lives one level up (at research_789/) and can contain year profiles.
KNOWLEDGE_ROOT = ROOT.parent / "knowledge"
KNOWLEDGE_DIR  = _resolve_knowledge_dir(ROOT)
KNOWLEDGE_PROFILE = KNOWLEDGE_DIR.name if KNOWLEDGE_DIR != KNOWLEDGE_ROOT else "root"
OUTPUT_PROFILE = _slugify(os.getenv("OUTPUT_PROFILE", KNOWLEDGE_PROFILE))

IS_INTEL_MAC = platform.system() == "Darwin" and platform.machine().lower() in {"x86_64", "amd64"}
LOW_RESOURCE_MODE = _env_bool("LOW_RESOURCE_MODE", IS_INTEL_MAC)

# --------------------------------------------------------------------------
# Primary statutory source (USLM XML, Title 26 IRC)
# --------------------------------------------------------------------------
USC26_XML = KNOWLEDGE_DIR / "usc26.xml"

# --------------------------------------------------------------------------
# IRS XML sources to skip during ingestion.
# p519: U.S. tax guide for aliens (non-resident scope, excluded).
# i1040nr: Form 1040-NR instructions (non-resident filers, excluded).
# --------------------------------------------------------------------------
EXCLUDED_SOURCES: set[str] = {"i1040nr", "p519"}

# --------------------------------------------------------------------------
# Pipeline outputs
# --------------------------------------------------------------------------
DATA_PROCESSED_BASE    = ROOT / "data" / "processed"
PROFILED_OUTPUTS       = _env_bool("PROFILED_OUTPUTS", True)
if PROFILED_OUTPUTS and OUTPUT_PROFILE != "root":
	DATA_PROCESSED = DATA_PROCESSED_BASE / OUTPUT_PROFILE
else:
	DATA_PROCESSED = DATA_PROCESSED_BASE

CHUNKS_FILE            = DATA_PROCESSED / "chunks.json"
VECTOR_CONTENT_INDEX   = DATA_PROCESSED / "vector_content.faiss"
VECTOR_SECTIONID_INDEX = DATA_PROCESSED / "vector_sectionid.faiss"
VECTOR_META_FILE       = DATA_PROCESSED / "vector_meta.json"
GRAPH_FILE             = DATA_PROCESSED / "graph.graphml"
COMMUNITY_FILE         = DATA_PROCESSED / "communities.json"
GRAPH_AUDIT_FILE       = DATA_PROCESSED / "graph_audit.json"
BUILD_MANIFEST_FILE    = DATA_PROCESSED / "build_manifest.json"
BUILD_CACHE_FILE       = DATA_PROCESSED / "build_cache.json"
BUILD_CACHE_ENABLED    = _env_bool("BUILD_CACHE_ENABLED", True)

# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
MAX_CHUNK_CHARS = 2000
SPACY_MODEL     = "en_core_web_sm"
ENABLE_SENTENCE_SPLIT = _env_bool("ENABLE_SENTENCE_SPLIT", not LOW_RESOURCE_MODE)

# PDF parser tuning (used by irs_pdf_parser.py)
PDF_SECTION_MIN_CHARS = _env_int("PDF_SECTION_MIN_CHARS", 120)
PDF_SECTION_MAX_CHARS = _env_int("PDF_SECTION_MAX_CHARS", 1400)
PDF_SECTION_OVERLAP_CHARS = _env_int("PDF_SECTION_OVERLAP_CHARS", 180)
PDF_MAX_SECTIONS_PER_PAGE = _env_int("PDF_MAX_SECTIONS_PER_PAGE", 20)

# --------------------------------------------------------------------------
# Embedding and retrieval
# --------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv(
	"EMBEDDING_MODEL",
	"all-MiniLM-L6-v2" if LOW_RESOURCE_MODE else "all-mpnet-base-v2",
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")
_DEFAULT_EMBEDDING_THREADS = (
	1 if LOW_RESOURCE_MODE else min(8, max(1, (os.cpu_count() or 1) - 1))
)
EMBEDDING_THREADS = max(1, _env_int("EMBEDDING_THREADS", _DEFAULT_EMBEDDING_THREADS))
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32 if LOW_RESOURCE_MODE else 16)
EMBEDDING_ENCODE_CHUNK_SIZE = _env_int("EMBEDDING_ENCODE_CHUNK_SIZE", 128 if LOW_RESOURCE_MODE else 256)
EMBEDDING_SHOW_PROGRESS = _env_bool("EMBEDDING_SHOW_PROGRESS", True)
DUAL_VECTOR_EMBEDDING = _env_bool("DUAL_VECTOR_EMBEDDING", not LOW_RESOURCE_MODE)
# Retrieval-side FAISS GPU acceleration (CUDA only). Falls back to CPU when unavailable.
FAISS_USE_GPU = _env_bool("FAISS_USE_GPU", True)
FAISS_GPU_DEVICE = max(0, _env_int("FAISS_GPU_DEVICE", 0))
FAISS_GPU_USE_FLOAT16 = _env_bool("FAISS_GPU_USE_FLOAT16", False)
# Repeated query texts are common in eval runs (vector + hybrid); cache encoded query vectors.
VECTOR_QUERY_EMBED_CACHE_SIZE = max(0, _env_int("VECTOR_QUERY_EMBED_CACHE_SIZE", 1024))
# Cache full vector query results for repeated eval queries (vector + hybrid).
VECTOR_QUERY_RESULT_CACHE_SIZE = max(0, _env_int("VECTOR_QUERY_RESULT_CACHE_SIZE", 1024))
# Search backend selection: auto | faiss | torch
# auto: prefer FAISS GPU when available; otherwise use torch matmul on CUDA; fallback to FAISS CPU.
VECTOR_SEARCH_BACKEND = os.getenv("VECTOR_SEARCH_BACKEND", "auto").strip().lower()
VECTOR_TORCH_FP16 = _env_bool("VECTOR_TORCH_FP16", True)
# Skip section-id index search when dual-vector section embeddings are disabled.
VECTOR_SEARCH_SECTIONID = _env_bool("VECTOR_SEARCH_SECTIONID", DUAL_VECTOR_EMBEDDING)
TOP_K_VECTOR    = 10
BFS_DEPTH       = 2
# Keep SARA retrieval query focused by default to reduce graph/vector noise.
SARA_APPEND_TEXT_CONTEXT_TO_RETRIEVAL = _env_bool(
	"SARA_APPEND_TEXT_CONTEXT_TO_RETRIEVAL", False
)
# Trim long retrieval excerpts before adding to prompts.
PROMPT_EXCERPT_MAX_CHARS = _env_int("PROMPT_EXCERPT_MAX_CHARS", 1000)

# --------------------------------------------------------------------------
# Graph retrieval scoring constants
# Edge-type priorities used when ordering BFS frontier neighbors.
# --------------------------------------------------------------------------
GRAPH_EDGE_WEIGHT_XREF:      float = 3.0
GRAPH_EDGE_WEIGHT_COVERAGE:  float = 1.8
GRAPH_EDGE_WEIGHT_HIERARCHY: float = 2.2
GRAPH_EDGE_WEIGHT_DEFAULT:   float = 1.0
# Per-term overlap contribution added to each neighbor's priority score.
GRAPH_OVERLAP_WEIGHT: float = 0.30
# Small authority boost for USC26 nodes (primary statute).
GRAPH_USC_BOOST: float = 0.15
# Entry-node raw-score filter thresholds (before [0.60,1.00] normalization).
GRAPH_ENTRY_THRESHOLD_BASE:    float = 3.5
GRAPH_ENTRY_THRESHOLD_SECTION: float = 5.5  # raised when query has explicit § refs
# Caps and cache for graph query speed in evaluation loops.
GRAPH_MAX_ENTRY_NODES = max(1, _env_int("GRAPH_MAX_ENTRY_NODES", 24))
GRAPH_MAX_NEIGHBORS_PER_NODE = max(1, _env_int("GRAPH_MAX_NEIGHBORS_PER_NODE", 20))
GRAPH_QUERY_CACHE_SIZE = max(0, _env_int("GRAPH_QUERY_CACHE_SIZE", 1024))
# Penalize lower-confidence inferred/fallback coverage edges during traversal.
GRAPH_COVERAGE_PENALTY_INFERRED = _env_float("GRAPH_COVERAGE_PENALTY_INFERRED", 0.25)
GRAPH_COVERAGE_PENALTY_FALLBACK = _env_float("GRAPH_COVERAGE_PENALTY_FALLBACK", 0.45)

# --------------------------------------------------------------------------
# Hybrid retrieval blending
# score = alpha * vector_score + (1 - alpha) * graph_score
# --------------------------------------------------------------------------
# Default: slightly vector-weighted for broad/semantic queries.
HYBRID_ALPHA_DEFAULT: float = 0.6
# When the query cites an explicit IRC § reference, lean on the graph more.
HYBRID_ALPHA_SECTION_REF: float = 0.5
# Cache merged retrieval results by (mode, query, k, depth).
HYBRID_QUERY_CACHE_SIZE = max(0, _env_int("HYBRID_QUERY_CACHE_SIZE", 1024))
# Normalize vector/graph scores before blending to reduce channel scale mismatch.
HYBRID_SCORE_NORMALIZE = _env_bool("HYBRID_SCORE_NORMALIZE", True)

# --------------------------------------------------------------------------
# Graph linking and audit
# --------------------------------------------------------------------------
LINKER_MAX_NODES_PER_SOURCE       = 3
LINKER_MAX_XPUB_REP_NODES         = 2
LINKER_MAX_TARGET_SECTIONS        = 8
LINKER_MIN_SECTION_REF_SUPPORT    = 3
LINKER_MIN_CROSS_PUB_SUPPORT      = 3
LINKER_ENABLE_FALLBACK_CONNECTIVITY = _env_bool("LINKER_ENABLE_FALLBACK_CONNECTIVITY", True)
LINKER_FALLBACK_MIN_XREF_COVERAGE = _env_int("LINKER_FALLBACK_MIN_XREF_COVERAGE", 1)

# Enforce graph coverage checks during build_pipeline.py
GRAPH_AUDIT_STRICT                = True
GRAPH_AUDIT_MIN_XREF_COVERAGE_EDGES = 1

# --------------------------------------------------------------------------
# LLMs
# --------------------------------------------------------------------------
CLAUDE_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT_SECONDS = _env_int("OLLAMA_TIMEOUT_SECONDS", 180)
OLLAMA_MAX_RETRIES = _env_int("OLLAMA_MAX_RETRIES", 2)
OLLAMA_RETRY_BACKOFF_SECONDS = _env_int("OLLAMA_RETRY_BACKOFF_SECONDS", 2)
OLLAMA_THINK = _env_bool("OLLAMA_THINK", False)
# Optional Ollama per-request generation controls.
# Use 0 / negative sentinel defaults to leave the daemon model defaults unchanged.
OLLAMA_NUM_CTX     = max(0, _env_int("OLLAMA_NUM_CTX", 0))
OLLAMA_NUM_PREDICT = _env_int("OLLAMA_NUM_PREDICT", 0)
OLLAMA_TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", -1.0)
OLLAMA_TOP_P       = _env_float("OLLAMA_TOP_P", -1.0)
# Number of GPU layers to offload (-1 = Ollama default; 99 = push all layers to GPU).
# Set to 99 to maximise GPU usage. Ollama caps at the model's actual layer count.
OLLAMA_NUM_GPU = _env_int("OLLAMA_NUM_GPU", 99)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
# How many cases to run in parallel. 1 = sequential.
# Set > 1 only when OLLAMA_NUM_PARALLEL is also raised server-side (env var
# on the Ollama process), otherwise concurrent requests queue and offer no gain.
EVAL_CONCURRENCY = max(1, _env_int("EVAL_CONCURRENCY", 1))

# Per-answer-type token caps for SARA.
# All types now require a three-step reasoning chain (Rule → Facts → Reasoning)
# before the Final Answer, so label cases need more room than a bare label would.
SARA_MAX_TOKENS_LABEL   = _env_int("SARA_MAX_TOKENS_LABEL",    600)
SARA_MAX_TOKENS_NUMERIC = _env_int("SARA_MAX_TOKENS_NUMERIC",  900)
SARA_MAX_TOKENS_STRING  = _env_int("SARA_MAX_TOKENS_STRING",   600)
SARA_MAX_TOKENS_DEFAULT = _env_int("SARA_MAX_TOKENS_DEFAULT",  800)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
