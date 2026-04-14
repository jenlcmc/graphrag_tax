"""Build the full data pipeline and save all artifacts to data/processed.

Run this once before any retrieval or evaluation. Steps:
  1. Parse Title 26 XML (Subtitle A only)
  2. Auto-discover and parse IRS XML/PDF publications from selected knowledge profile
  3. Normalize: extract USC and IRS cross-references from explanation text
  4. Chunk: split oversized chunks at sentence boundaries
  5. Save chunks.json
  6. Build/save FAISS vector indexes
  7. Build/save NetworkX graph + communities + audit report

This script supports incremental caching. If source files and relevant settings
are unchanged, chunking/indexing steps are skipped and existing artifacts are
reused.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config as cfg
from src.indexing.graph_audit import audit_graph_coverage, save_audit_report
from src.indexing.graph_index import GraphIndex
from src.indexing.vector_index import VectorIndex
from src.ingestion.irs_pdf_parser import parse as parse_irs_pdf
from src.ingestion.irs_xml_parser import parse as parse_irs_xml
from src.ingestion.sara_source_parser import parse as parse_sara_source
from src.ingestion.usc26_parser import parse as parse_usc26
from src.preprocessing.chunker import apply_to_all
from src.preprocessing.normalizer import annotate_chunks, load_nlp


_CACHE_VERSION = 1
_CHUNK_STAGE_VERSION = "chunk_stage_v2"
_VECTOR_STAGE_VERSION = "vector_stage_v1"
_GRAPH_STAGE_VERSION = "graph_stage_v2"


def _pdf_source_key(pdf_path: Path) -> str | None:
    """Return normalized source key from IRS filename, or None if unknown.

    Examples:
      p17--2017.pdf      -> p17
      i1040sc--2017.pdf  -> i1040sc
    """
    stem = pdf_path.stem.lower()
    if "--" in stem:
        stem = stem.split("--", 1)[0]
    stem = re.sub(r"[^a-z0-9]", "", stem)
    if not stem:
        return None
    if re.fullmatch(r"(?:p|i|f)\d+[a-z0-9]*", stem) is None:
        return None
    return stem


def discover_irs_sources(knowledge_dir: Path, excluded: set[str]) -> list[tuple[str, Path, str]]:
    """Return (source_key, path, format) for IRS sources in selected knowledge_dir.

    Supported layouts:
      1) XML subdirectories: knowledge/<source>/<source>.xml
      2) Flat XML files:     knowledge/<source>.xml
      3) Flat PDF files:     knowledge/<source>--<year>.pdf

    XML is preferred over PDF when both are present for the same source.
    """
    if not knowledge_dir.exists():
        return []

    sources: dict[str, tuple[Path, str]] = {}

    # XML subdirectory layout.
    for subdir in sorted(knowledge_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name.lower()
        xml_file = subdir / f"{subdir.name}.xml"
        if name in excluded:
            continue
        if xml_file.exists():
            sources[name] = (xml_file, "xml")

    # Flat XML files.
    for xml_file in sorted(knowledge_dir.glob("*.xml")):
        name = xml_file.stem.lower()
        if name == "usc26" or name in excluded:
            continue
        sources.setdefault(name, (xml_file, "xml"))

    # Flat PDF files.
    for pdf_file in sorted(knowledge_dir.glob("*.pdf")):
        name = _pdf_source_key(pdf_file)
        if name is None or name in excluded:
            continue
        sources.setdefault(name, (pdf_file, "pdf"))

    return [(source, path, fmt) for source, (path, fmt) in sorted(sources.items())]


def discover_sara_source_files(knowledge_dir: Path) -> list[Path]:
    """Return SARA statute source files under knowledge/<profile>/source/.

    Files are plain text sections (for example: section151, section63).
    """
    source_dir = knowledge_dir / "source"
    if not source_dir.exists() or not source_dir.is_dir():
        return []

    return [path for path in sorted(source_dir.iterdir()) if path.is_file()]


def _sha256_json(data: dict) -> str:
    blob = json.dumps(data, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _file_snapshot(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _source_snapshot(
    usc_path: Path,
    irs_sources: list[tuple[str, Path, str]],
    sara_source_files: list[Path],
) -> list[dict]:
    snapshot = [{"source": "usc26", "format": "xml", **_file_snapshot(usc_path)}]
    for source, path, fmt in irs_sources:
        snapshot.append({"source": source, "format": fmt, **_file_snapshot(path)})
    for path in sara_source_files:
        snapshot.append(
            {
                "source": "sara_source",
                "format": "text",
                **_file_snapshot(path),
            }
        )
    return snapshot


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        cache = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return {}
    if cache.get("version") != _CACHE_VERSION:
        return {}
    return cache


def _all_files_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _count_by_source(chunks: list[dict]) -> dict[str, int]:
    counter = Counter(chunk.get("source", "") for chunk in chunks)
    return {key: counter[key] for key in sorted(counter) if key}


def _build_chunks(
    irs_sources: list[tuple[str, Path, str]],
    sara_source_files: list[Path],
) -> tuple[list[dict], dict[str, int]]:
    print("Parsing Title 26 XML (Subtitle A)...")
    usc_chunks = parse_usc26(cfg.USC26_XML, cfg.MAX_CHUNK_CHARS)
    print(f"  {len(usc_chunks)} statute chunks")

    print("Discovering IRS sources in selected knowledge profile...")
    print(f"  Found {len(irs_sources)} IRS sources: {[s for s, _, _ in irs_sources]}")

    irs_chunks: list[dict] = []
    per_source_counts: dict[str, int] = {}
    for source_key, source_path, source_format in irs_sources:
        try:
            if source_format == "xml":
                chunks = parse_irs_xml(source_path, source_key)
            else:
                chunks = parse_irs_pdf(source_path, source_key)
            irs_chunks.extend(chunks)
            per_source_counts[source_key] = len(chunks)
            print(f"    {source_key} ({source_format}): {len(chunks)} chunks")
        except Exception as exc:
            print(f"    WARNING: failed to parse {source_path.name} - {exc}")

    print(f"  {len(irs_chunks)} total IRS explanation chunks")

    sara_chunks: list[dict] = []
    if sara_source_files:
        print("Parsing SARA statute source files from knowledge/source...")
        sara_source_dir = sara_source_files[0].parent
        sara_chunks = parse_sara_source(sara_source_dir, source="sara_source")
        per_source_counts["sara_source"] = len(sara_chunks)
        print(
            "    sara_source (text): "
            f"{len(sara_chunks)} chunks from {len(sara_source_files)} files"
        )

    all_chunks = usc_chunks + irs_chunks + sara_chunks

    print("Loading spaCy model and annotating cross-references...")
    nlp = load_nlp(cfg.SPACY_MODEL)
    all_chunks = annotate_chunks(all_chunks, nlp)

    if cfg.ENABLE_SENTENCE_SPLIT:
        print("Splitting oversized chunks at sentence boundaries...")
        all_chunks = apply_to_all(all_chunks, nlp, cfg.MAX_CHUNK_CHARS)
        print(f"  {len(all_chunks)} total chunks after splitting")
    else:
        print("Skipping sentence splitting (low-resource mode)")
        print(f"  {len(all_chunks)} total chunks")

    return all_chunks, per_source_counts


def _graph_build_and_audit(all_chunks: list[dict]) -> dict:
    print("Building GraphRAG knowledge graph with community detection...")
    graph_idx = GraphIndex()
    graph_idx.build(all_chunks)
    graph_idx.save(cfg.GRAPH_FILE, cfg.COMMUNITY_FILE)

    print("Running graph coverage audit...")
    report = audit_graph_coverage(
        graph_idx.graph,
        min_xref_coverage_edges=cfg.GRAPH_AUDIT_MIN_XREF_COVERAGE_EDGES,
    )
    save_audit_report(report, cfg.GRAPH_AUDIT_FILE)

    if report["ok"]:
        print(
            "Graph audit passed: "
            f"{report['n_sources']} sources are connected with sufficient "
            "xref/coverage edges"
        )
    else:
        print(
            "Graph audit found "
            f"{len(report['issues'])} issue(s). "
            f"Report saved -> {cfg.GRAPH_AUDIT_FILE}"
        )
        for issue in report["issues"][:20]:
            print(f"  - {issue}")
        if len(report["issues"]) > 20:
            print("  - ...")

        if cfg.GRAPH_AUDIT_STRICT:
            raise RuntimeError(
                "Graph audit failed in strict mode. "
                f"See {cfg.GRAPH_AUDIT_FILE} for full details."
            )

    return report


def _write_manifest(
    all_chunks: list[dict],
    irs_sources: list[tuple[str, Path, str]],
    sara_source_files: list[Path],
    per_source_counts: dict[str, int],
    cache_hits: dict[str, bool],
    source_fingerprint: str,
    chunks_hash: str,
    vector_fingerprint: str,
    graph_fingerprint: str,
) -> None:
    format_counts = Counter(fmt for _, _, fmt in irs_sources)
    if sara_source_files:
        format_counts["text"] += len(sara_source_files)
    source_counts = _count_by_source(all_chunks)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "knowledge_profile": cfg.KNOWLEDGE_PROFILE,
        "knowledge_dir": str(cfg.KNOWLEDGE_DIR),
        "output_profile": cfg.OUTPUT_PROFILE,
        "output_dir": str(cfg.DATA_PROCESSED),
        "n_chunks": len(all_chunks),
        "n_usc_chunks": source_counts.get("usc26", 0),
        "n_irs_chunks": len(all_chunks) - source_counts.get("usc26", 0),
        "source_count": len([key for key in source_counts if key != "usc26"]),
        "source_counts": source_counts,
        "ingest_source_formats": dict(format_counts),
        "irs_ingest_chunk_counts": per_source_counts,
        "cache_hits": cache_hits,
        "fingerprints": {
            "source": source_fingerprint,
            "chunks": chunks_hash,
            "vector_stage": vector_fingerprint,
            "graph_stage": graph_fingerprint,
        },
    }
    cfg.BUILD_MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))
    print(f"Saved build manifest -> {cfg.BUILD_MANIFEST_FILE}")


def main() -> None:
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"Using knowledge profile: {cfg.KNOWLEDGE_PROFILE}")
    print(f"Using knowledge directory: {cfg.KNOWLEDGE_DIR}")
    print(f"Using output directory : {cfg.DATA_PROCESSED}")

    if not cfg.USC26_XML.exists():
        raise FileNotFoundError(
            "USC XML not found at "
            f"{cfg.USC26_XML}. "
            "Set KNOWLEDGE_PROFILE to a folder containing usc26.xml."
        )

    irs_sources = discover_irs_sources(cfg.KNOWLEDGE_DIR, cfg.EXCLUDED_SOURCES)
    sara_source_files = discover_sara_source_files(cfg.KNOWLEDGE_DIR)

    source_snapshot = _source_snapshot(cfg.USC26_XML, irs_sources, sara_source_files)
    source_fingerprint = _sha256_json({"snapshot": source_snapshot})

    chunks_stage_fingerprint = _sha256_json(
        {
            "version": _CHUNK_STAGE_VERSION,
            "source_fingerprint": source_fingerprint,
            "max_chunk_chars": cfg.MAX_CHUNK_CHARS,
            "spacy_model": cfg.SPACY_MODEL,
            "enable_sentence_split": cfg.ENABLE_SENTENCE_SPLIT,
            "pdf_section_min_chars": cfg.PDF_SECTION_MIN_CHARS,
            "pdf_section_max_chars": cfg.PDF_SECTION_MAX_CHARS,
            "pdf_section_overlap_chars": cfg.PDF_SECTION_OVERLAP_CHARS,
            "pdf_max_sections_per_page": cfg.PDF_MAX_SECTIONS_PER_PAGE,
        }
    )

    cache = _load_cache(cfg.BUILD_CACHE_FILE) if cfg.BUILD_CACHE_ENABLED else {}
    cache_hits = {"chunks": False, "vector": False, "graph": False}

    per_source_counts: dict[str, int] = {}
    if (
        cfg.BUILD_CACHE_ENABLED
        and cache.get("chunks_stage_fingerprint") == chunks_stage_fingerprint
        and cfg.CHUNKS_FILE.exists()
    ):
        print("Chunk stage cache hit: reusing chunks.json")
        all_chunks = json.loads(cfg.CHUNKS_FILE.read_text())
        cache_hits["chunks"] = True
        per_source_counts = {
            source: count
            for source, count in _count_by_source(all_chunks).items()
            if source != "usc26"
        }
    else:
        print("Chunk stage cache miss: rebuilding chunks")
        all_chunks, per_source_counts = _build_chunks(irs_sources, sara_source_files)
        cfg.CHUNKS_FILE.write_text(json.dumps(all_chunks, indent=2))
        print(f"Saved chunks -> {cfg.CHUNKS_FILE}")

    chunks_hash = _sha256_file(cfg.CHUNKS_FILE)

    vector_fingerprint = _sha256_json(
        {
            "version": _VECTOR_STAGE_VERSION,
            "chunks_hash": chunks_hash,
            "embedding_model": cfg.EMBEDDING_MODEL,
            "embedding_device": cfg.EMBEDDING_DEVICE,
            "embedding_threads": cfg.EMBEDDING_THREADS,
            "embedding_batch_size": cfg.EMBEDDING_BATCH_SIZE,
            "embedding_encode_chunk_size": cfg.EMBEDDING_ENCODE_CHUNK_SIZE,
            "dual_vector_embedding": cfg.DUAL_VECTOR_EMBEDDING,
        }
    )

    vector_files = [
        cfg.VECTOR_CONTENT_INDEX,
        cfg.VECTOR_SECTIONID_INDEX,
        cfg.VECTOR_META_FILE,
    ]
    if (
        cfg.BUILD_CACHE_ENABLED
        and cache.get("vector_stage_fingerprint") == vector_fingerprint
        and _all_files_exist(vector_files)
    ):
        print("Vector stage cache hit: reusing FAISS indexes")
        cache_hits["vector"] = True
    else:
        print("Building FAISS vector index...")
        vector_idx = VectorIndex(cfg.EMBEDDING_MODEL)
        vector_idx.build(all_chunks)
        vector_idx.save(
            cfg.VECTOR_CONTENT_INDEX,
            cfg.VECTOR_SECTIONID_INDEX,
            cfg.VECTOR_META_FILE,
        )

    graph_fingerprint = _sha256_json(
        {
            "version": _GRAPH_STAGE_VERSION,
            "chunks_hash": chunks_hash,
            "graph_audit_min_xref_coverage": cfg.GRAPH_AUDIT_MIN_XREF_COVERAGE_EDGES,
            "linker_max_nodes_per_source": cfg.LINKER_MAX_NODES_PER_SOURCE,
            "linker_max_xpub_rep_nodes": cfg.LINKER_MAX_XPUB_REP_NODES,
            "linker_max_target_sections": cfg.LINKER_MAX_TARGET_SECTIONS,
            "linker_min_section_ref_support": cfg.LINKER_MIN_SECTION_REF_SUPPORT,
            "linker_min_cross_pub_support": cfg.LINKER_MIN_CROSS_PUB_SUPPORT,
            "linker_enable_fallback_connectivity": cfg.LINKER_ENABLE_FALLBACK_CONNECTIVITY,
            "linker_fallback_min_xref_coverage": cfg.LINKER_FALLBACK_MIN_XREF_COVERAGE,
        }
    )

    graph_files = [cfg.GRAPH_FILE, cfg.COMMUNITY_FILE, cfg.GRAPH_AUDIT_FILE]
    if (
        cfg.BUILD_CACHE_ENABLED
        and cache.get("graph_stage_fingerprint") == graph_fingerprint
        and _all_files_exist(graph_files)
    ):
        print("Graph stage cache hit: reusing graph, communities, and audit report")
        cache_hits["graph"] = True
        audit_report = json.loads(cfg.GRAPH_AUDIT_FILE.read_text())
        if not audit_report.get("ok") and cfg.GRAPH_AUDIT_STRICT:
            raise RuntimeError(
                "Cached graph audit is failing in strict mode. "
                f"See {cfg.GRAPH_AUDIT_FILE} for details."
            )
    else:
        audit_report = _graph_build_and_audit(all_chunks)

    if cfg.BUILD_CACHE_ENABLED:
        cache_payload = {
            "version": _CACHE_VERSION,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "knowledge_profile": cfg.KNOWLEDGE_PROFILE,
            "output_profile": cfg.OUTPUT_PROFILE,
            "chunks_stage_fingerprint": chunks_stage_fingerprint,
            "vector_stage_fingerprint": vector_fingerprint,
            "graph_stage_fingerprint": graph_fingerprint,
            "source_fingerprint": source_fingerprint,
            "chunks_hash": chunks_hash,
            "audit_ok": bool(audit_report.get("ok", False)),
        }
        cfg.BUILD_CACHE_FILE.write_text(json.dumps(cache_payload, indent=2))
        print(f"Saved build cache -> {cfg.BUILD_CACHE_FILE}")

    _write_manifest(
        all_chunks=all_chunks,
        irs_sources=irs_sources,
        sara_source_files=sara_source_files,
        per_source_counts=per_source_counts,
        cache_hits=cache_hits,
        source_fingerprint=source_fingerprint,
        chunks_hash=chunks_hash,
        vector_fingerprint=vector_fingerprint,
        graph_fingerprint=graph_fingerprint,
    )

    print("\nPipeline complete. Artifacts saved to:", cfg.DATA_PROCESSED)


if __name__ == "__main__":
    main()
