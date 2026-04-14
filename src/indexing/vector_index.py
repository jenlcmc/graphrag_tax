"""FAISS-based dual vector index.

Two separate indices are built per chunk (following de Oliveira Lima 2025):
  - content index:    embeds the full text of each chunk
  - section_id index: embeds only the canonical section identifier string

At query time, both indices are searched and results are unioned. The content
index handles semantic queries; the section_id index handles queries that
directly name a section (e.g. "401k" matching "26 USC §401(k)").

Files written:
  vector_content.faiss    - FAISS IndexFlatIP for content embeddings
  vector_sectionid.faiss  - FAISS IndexFlatIP for section_id embeddings
  vector_meta.json        - metadata list aligned by index position
"""

import os
import json
import warnings
import numpy as np
from pathlib import Path
from src import config as cfg

# Apply runtime controls before importing ML libraries.
os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
os.environ["HF_DATASETS_OFFLINE"] = os.environ.get("HF_DATASETS_OFFLINE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")
# Prevent duplicate OpenMP library crashes on Intel Mac and Windows with MKL.
os.environ["KMP_DUPLICATE_LIB_OK"] = os.environ.get("KMP_DUPLICATE_LIB_OK", "TRUE")
# Disable macOS Obj-C fork-safety check that causes segfaults when PyTorch
# inherits file descriptors across fork boundaries (Intel Mac, Python 3.10+).
# This env var is silently ignored on Linux and Windows.
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = os.environ.get(
    "OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES"
)
thread_count = str(cfg.EMBEDDING_THREADS)
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", thread_count)
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", thread_count)
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", thread_count)
os.environ["NUMEXPR_NUM_THREADS"] = os.environ.get("NUMEXPR_NUM_THREADS", thread_count)
os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ.get("VECLIB_MAXIMUM_THREADS", thread_count)

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Suppress FutureWarning from transformers 4.44 about clean_up_tokenization_spaces
warnings.filterwarnings(
    "ignore",
    message=".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
)


class VectorIndex:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        torch.set_num_threads(cfg.EMBEDDING_THREADS)
        try:
            torch.set_num_interop_threads(max(1, min(2, cfg.EMBEDDING_THREADS)))
        except RuntimeError:
            pass

        self.model          = SentenceTransformer(model_name, device=cfg.EMBEDDING_DEVICE)
        self.content_index  = None
        self.sectionid_index = None
        self.metadata       = []

    def build(self, chunks: list[dict]) -> None:
        texts       = [c["text"]       for c in chunks]
        section_ids = [c["section_id"] for c in chunks]

        print(f"Embedding {len(chunks)} chunks (content)...")
        content_vecs = self._encode_in_outer_chunks(texts, label="content")

        if cfg.DUAL_VECTOR_EMBEDDING:
            print(f"Embedding {len(chunks)} chunks (section_id)...")
            sectionid_vecs = self._encode_in_outer_chunks(section_ids, label="section_id")
        else:
            print("Low-resource mode: reusing content vectors for section_id index")
            sectionid_vecs = content_vecs

        import faiss

        dim = content_vecs.shape[1]
        self.content_index   = faiss.IndexFlatIP(dim)
        self.sectionid_index = faiss.IndexFlatIP(dim)

        self.content_index.add(content_vecs.astype(np.float32))
        self.sectionid_index.add(sectionid_vecs.astype(np.float32))

        self.metadata = [
            {
                "id":         c["id"],
                "section_id": c["section_id"],
                "title":      c["title"],
                "source":     c["source"],
                "text":       c["text"],
                "hierarchy":  c.get("hierarchy"),
                "parent_id":  c.get("parent_id"),
                "cross_refs": c["cross_refs"],
                "source_file": c.get("source_file"),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "chunk_method": c.get("chunk_method"),
                "metadata": c.get("metadata", {}),
            }
            for c in chunks
        ]

    def save(
        self,
        content_path: Path,
        sectionid_path: Path,
        meta_path: Path,
    ) -> None:
        import faiss

        faiss.write_index(self.content_index,   str(content_path))
        faiss.write_index(self.sectionid_index, str(sectionid_path))
        meta_path.write_text(json.dumps(self.metadata, indent=2))
        print(f"Saved {len(self.metadata)} vectors to {content_path.parent}")

    def load(
        self,
        content_path: Path,
        sectionid_path: Path,
        meta_path: Path,
    ) -> None:
        import faiss

        self.content_index   = faiss.read_index(str(content_path))
        self.sectionid_index = faiss.read_index(str(sectionid_path))
        self.metadata        = json.loads(meta_path.read_text())

    def query(self, text: str, k: int = 10) -> list[dict]:
        """Search both indices and return the union of top-k results.

        Results are scored by the maximum of their content and section_id
        cosine similarities.
        """
        vec = self.model.encode([text], normalize_embeddings=True).astype(np.float32)

        c_scores, c_ids = self.content_index.search(vec, k)
        s_scores, s_ids = self.sectionid_index.search(vec, k)

        # Merge: keep best score per chunk index
        scores = {}
        for score, idx in zip(c_scores[0], c_ids[0]):
            scores[int(idx)] = float(score)
        for score, idx in zip(s_scores[0], s_ids[0]):
            idx = int(idx)
            scores[idx] = max(scores.get(idx, 0.0), float(score))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {**self.metadata[idx], "vector_score": score}
            for idx, score in ranked
        ]

    def _encode_in_outer_chunks(self, texts: list[str], label: str) -> np.ndarray:
        """Encode in bounded outer chunks for stability on low-resource machines."""
        if not texts:
            dim = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)

        outer = max(cfg.EMBEDDING_BATCH_SIZE, cfg.EMBEDDING_ENCODE_CHUNK_SIZE)
        starts = range(0, len(texts), outer)

        if cfg.EMBEDDING_SHOW_PROGRESS:
            total = (len(texts) + outer - 1) // outer
            starts_iter = tqdm(starts, total=total, desc=f"Outer chunks ({label})")
        else:
            starts_iter = starts

        vectors: list[np.ndarray] = []
        for start in starts_iter:
            batch_texts = texts[start : start + outer]
            # no_grad avoids gradient tracking overhead; convert_to_numpy=True
            # keeps everything in-process (no DataLoader worker forks), which
            # prevents the semaphore-leak segfault on Intel Mac and Windows.
            with torch.no_grad():
                encoded = self.model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=cfg.EMBEDDING_BATCH_SIZE,
                    convert_to_numpy=True,
                )
            vectors.append(np.asarray(encoded, dtype=np.float32))

        return np.vstack(vectors)
