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
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_THREADS = max(1, _env_int("EMBEDDING_THREADS", 1))
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32 if LOW_RESOURCE_MODE else 16)
EMBEDDING_ENCODE_CHUNK_SIZE = _env_int("EMBEDDING_ENCODE_CHUNK_SIZE", 128 if LOW_RESOURCE_MODE else 256)
EMBEDDING_SHOW_PROGRESS = _env_bool("EMBEDDING_SHOW_PROGRESS", True)
DUAL_VECTOR_EMBEDDING = _env_bool("DUAL_VECTOR_EMBEDDING", not LOW_RESOURCE_MODE)
TOP_K_VECTOR    = 10
BFS_DEPTH       = 2

# --------------------------------------------------------------------------
# Graph retrieval scoring constants
# Edge-type priorities used when ordering BFS frontier neighbors.
# --------------------------------------------------------------------------
GRAPH_EDGE_WEIGHT_XREF:      float = 3.0
GRAPH_EDGE_WEIGHT_COVERAGE:  float = 2.5
GRAPH_EDGE_WEIGHT_HIERARCHY: float = 1.8
GRAPH_EDGE_WEIGHT_DEFAULT:   float = 1.0
# Per-term overlap contribution added to each neighbor's priority score.
GRAPH_OVERLAP_WEIGHT: float = 0.20
# Small authority boost for USC26 nodes (primary statute).
GRAPH_USC_BOOST: float = 0.15
# Entry-node raw-score filter thresholds (before [0.60,1.00] normalization).
GRAPH_ENTRY_THRESHOLD_BASE:    float = 3.0
GRAPH_ENTRY_THRESHOLD_SECTION: float = 5.0  # raised when query has explicit § refs

# --------------------------------------------------------------------------
# Hybrid retrieval blending
# score = alpha * vector_score + (1 - alpha) * graph_score
# --------------------------------------------------------------------------
# Default: slightly vector-weighted for broad/semantic queries.
HYBRID_ALPHA_DEFAULT: float = 0.6
# When the query cites an explicit IRC § reference, lean on the graph more.
HYBRID_ALPHA_SECTION_REF: float = 0.35

# --------------------------------------------------------------------------
# Graph linking and audit
# --------------------------------------------------------------------------
LINKER_MAX_NODES_PER_SOURCE       = 3
LINKER_MAX_XPUB_REP_NODES         = 2
LINKER_MAX_TARGET_SECTIONS        = 8
LINKER_MIN_SECTION_REF_SUPPORT    = 2
LINKER_MIN_CROSS_PUB_SUPPORT      = 2
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

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
