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
from collections import OrderedDict
from pathlib import Path
from src import config as cfg

# Apply runtime controls before importing ML libraries.
# Default to online mode so first-time model downloads can succeed.
# Set HF_HUB_OFFLINE=1 (or TRANSFORMERS_OFFLINE=1) to force offline execution.
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "0")
os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get(
    "TRANSFORMERS_OFFLINE", os.environ["HF_HUB_OFFLINE"]
)
os.environ["HF_DATASETS_OFFLINE"] = os.environ.get(
    "HF_DATASETS_OFFLINE", os.environ["HF_HUB_OFFLINE"]
)
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

        resolved_device = self._resolve_device(cfg.EMBEDDING_DEVICE)
        if resolved_device != cfg.EMBEDDING_DEVICE:
            print(
                f"Embedding device resolved: requested '{cfg.EMBEDDING_DEVICE}' -> using '{resolved_device}'"
            )

        self.embedding_device = resolved_device
        self.model          = SentenceTransformer(model_name, device=resolved_device)
        self.content_index  = None
        self.sectionid_index = None
        self.metadata       = []
        self._faiss = None
        self._gpu_resources = None
        self._faiss_gpu_enabled = False
        self._warned_no_gpu_faiss = False
        self._warned_no_torch_search = False
        self._logged_torch_backend = False
        self._torch_content_matrix: torch.Tensor | None = None
        self._torch_sectionid_matrix: torch.Tensor | None = None
        self._query_embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_result_cache: OrderedDict[tuple[str, int, bool], list[tuple[int, float]]] = OrderedDict()

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        requested = (requested_device or "auto").strip().lower()

        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend and mps_backend.is_available():
                return "mps"
            return "cpu"

        if requested == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but unavailable; falling back to CPU.")
            return "cpu"

        if requested == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if not (mps_backend and mps_backend.is_available()):
                print("MPS requested but unavailable; falling back to CPU.")
                return "cpu"

        return requested

    def build(self, chunks: list[dict]) -> None:
        self._query_embedding_cache.clear()
        self._query_result_cache.clear()
        self._clear_torch_matrices()
        self._logged_torch_backend = False

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

        faiss = self._faiss_module()

        dim = content_vecs.shape[1]
        self.content_index   = faiss.IndexFlatIP(dim)
        self.sectionid_index = faiss.IndexFlatIP(dim)

        self.content_index.add(content_vecs.astype(np.float32))
        self.sectionid_index.add(sectionid_vecs.astype(np.float32))
        self._maybe_enable_faiss_gpu("build")
        self._prepare_torch_matrices(content_vecs, sectionid_vecs)

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
        faiss = self._faiss_module()

        content_index = self._gpu_to_cpu_index(faiss, self.content_index)
        sectionid_index = self._gpu_to_cpu_index(faiss, self.sectionid_index)

        faiss.write_index(content_index, str(content_path))
        faiss.write_index(sectionid_index, str(sectionid_path))
        meta_path.write_text(json.dumps(self.metadata, indent=2))
        print(f"Saved {len(self.metadata)} vectors to {content_path.parent}")

    def load(
        self,
        content_path: Path,
        sectionid_path: Path,
        meta_path: Path,
    ) -> None:
        self._query_embedding_cache.clear()
        self._query_result_cache.clear()
        self._clear_torch_matrices()
        self._logged_torch_backend = False

        faiss = self._faiss_module()

        self.content_index   = faiss.read_index(str(content_path))
        self.sectionid_index = faiss.read_index(str(sectionid_path))
        self.metadata        = json.loads(meta_path.read_text())
        self._faiss_gpu_enabled = False
        self._maybe_enable_faiss_gpu("load")
        self._prepare_torch_matrices()

    def query(self, text: str, k: int = 10) -> list[dict]:
        """Search both indices and return the union of top-k results.

        Results are scored by the maximum of their content and section_id
        cosine similarities.
        """
        use_sectionid = bool(cfg.VECTOR_SEARCH_SECTIONID)
        cache_key = (text, int(k), use_sectionid)
        cached = self._cache_get_result(cache_key)
        if cached is not None:
            return [{**self.metadata[idx], "vector_score": score} for idx, score in cached]

        vec = self._encode_query(text)

        if self._should_use_torch_backend():
            if self._torch_content_matrix is None or (
                use_sectionid and self._torch_sectionid_matrix is None
            ):
                self._prepare_torch_matrices()
            scores = self._search_with_torch(vec, k, use_sectionid)
            if not scores:
                scores = self._search_with_faiss(vec, k, use_sectionid)
        else:
            scores = self._search_with_faiss(vec, k, use_sectionid)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self._cache_set_result(cache_key, ranked)
        return [
            {**self.metadata[idx], "vector_score": score}
            for idx, score in ranked
        ]

    def _effective_search_backend(self) -> str:
        backend = (cfg.VECTOR_SEARCH_BACKEND or "auto").strip().lower()
        if backend in {"auto", "faiss", "torch"}:
            return backend
        return "auto"

    def _should_use_torch_backend(self) -> bool:
        backend = self._effective_search_backend()
        if backend == "faiss":
            return False

        if self.embedding_device != "cuda" or not torch.cuda.is_available():
            if backend == "torch" and not self._warned_no_torch_search:
                print(
                    "VECTOR_SEARCH_BACKEND=torch requested but CUDA is unavailable; "
                    "falling back to FAISS."
                )
                self._warned_no_torch_search = True
            return False

        if backend == "torch":
            return True

        return bool(cfg.FAISS_USE_GPU) and not self._faiss_gpu_enabled

    def _clear_torch_matrices(self) -> None:
        self._torch_content_matrix = None
        self._torch_sectionid_matrix = None

    def _prepare_torch_matrices(
        self,
        content_vecs: np.ndarray | None = None,
        sectionid_vecs: np.ndarray | None = None,
    ) -> None:
        if not self._should_use_torch_backend():
            self._clear_torch_matrices()
            return

        dtype = torch.float16 if cfg.VECTOR_TORCH_FP16 else torch.float32
        device = torch.device("cuda")

        try:
            if content_vecs is None:
                content_vecs = self._extract_index_vectors(self.content_index)
            self._torch_content_matrix = torch.as_tensor(
                content_vecs,
                device=device,
                dtype=dtype,
            ).contiguous()

            if cfg.VECTOR_SEARCH_SECTIONID and self.sectionid_index is not None:
                if sectionid_vecs is None:
                    sectionid_vecs = self._extract_index_vectors(self.sectionid_index)
                self._torch_sectionid_matrix = torch.as_tensor(
                    sectionid_vecs,
                    device=device,
                    dtype=dtype,
                ).contiguous()
            else:
                self._torch_sectionid_matrix = None

            if not self._logged_torch_backend:
                precision = "float16" if cfg.VECTOR_TORCH_FP16 else "float32"
                print(f"Vector search backend: torch CUDA ({precision}).")
                self._logged_torch_backend = True
        except Exception as exc:
            if not self._warned_no_torch_search:
                print(f"Torch CUDA vector search unavailable; falling back to FAISS. Reason: {exc}")
                self._warned_no_torch_search = True
            self._clear_torch_matrices()

    def _extract_index_vectors(self, index) -> np.ndarray:
        faiss = self._faiss_module()
        cpu_index = self._gpu_to_cpu_index(faiss, index)
        total = int(getattr(cpu_index, "ntotal", 0))
        dim = int(getattr(cpu_index, "d", 0))

        if total <= 0:
            return np.empty((0, dim), dtype=np.float32)

        if hasattr(cpu_index, "reconstruct_n"):
            vectors = cpu_index.reconstruct_n(0, total)
            return np.asarray(vectors, dtype=np.float32)

        vectors = [cpu_index.reconstruct(i) for i in range(total)]
        return np.asarray(vectors, dtype=np.float32)

    def _search_with_faiss(
        self,
        vec: np.ndarray,
        k: int,
        use_sectionid: bool,
    ) -> dict[int, float]:
        c_scores, c_ids = self.content_index.search(vec, k)

        scores: dict[int, float] = {}
        for score, idx in zip(c_scores[0], c_ids[0]):
            idx = int(idx)
            if idx < 0:
                continue
            scores[idx] = float(score)

        if use_sectionid:
            s_scores, s_ids = self.sectionid_index.search(vec, k)
            for score, idx in zip(s_scores[0], s_ids[0]):
                idx = int(idx)
                if idx < 0:
                    continue
                scores[idx] = max(scores.get(idx, 0.0), float(score))

        return scores

    def _search_with_torch(
        self,
        vec: np.ndarray,
        k: int,
        use_sectionid: bool,
    ) -> dict[int, float]:
        if self._torch_content_matrix is None:
            return {}

        query = torch.from_numpy(vec).to(
            device=self._torch_content_matrix.device,
            dtype=self._torch_content_matrix.dtype,
        )
        query = query.squeeze(0).contiguous()

        scores: dict[int, float] = {}

        with torch.no_grad():
            content_scores = torch.matmul(self._torch_content_matrix, query)
            content_k = min(max(1, int(k)), int(content_scores.shape[0]))
            if content_k > 0:
                c_values, c_indices = torch.topk(content_scores, k=content_k, largest=True, sorted=True)
                for score, idx in zip(c_values.tolist(), c_indices.tolist()):
                    scores[int(idx)] = float(score)

            if use_sectionid and self._torch_sectionid_matrix is not None:
                section_scores = torch.matmul(self._torch_sectionid_matrix, query)
                section_k = min(max(1, int(k)), int(section_scores.shape[0]))
                if section_k > 0:
                    s_values, s_indices = torch.topk(section_scores, k=section_k, largest=True, sorted=True)
                    for score, idx in zip(s_values.tolist(), s_indices.tolist()):
                        idx = int(idx)
                        scores[idx] = max(scores.get(idx, 0.0), float(score))

        return scores

    def _faiss_module(self):
        if self._faiss is not None:
            return self._faiss

        import faiss

        if hasattr(faiss, "omp_set_num_threads"):
            try:
                faiss.omp_set_num_threads(cfg.EMBEDDING_THREADS)
            except Exception:
                pass

        self._faiss = faiss
        return faiss

    def _maybe_enable_faiss_gpu(self, where: str) -> None:
        if self.content_index is None or self.sectionid_index is None:
            return
        if not cfg.FAISS_USE_GPU:
            return
        if self._faiss_gpu_enabled:
            return
        if not torch.cuda.is_available():
            return

        faiss = self._faiss_module()
        if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
            if not self._warned_no_gpu_faiss:
                print("FAISS GPU is not available in this faiss build; using CPU indices.")
                self._warned_no_gpu_faiss = True
            return

        device = max(0, int(cfg.FAISS_GPU_DEVICE))
        cloner_options = None
        if hasattr(faiss, "GpuClonerOptions"):
            cloner_options = faiss.GpuClonerOptions()
            cloner_options.useFloat16 = bool(cfg.FAISS_GPU_USE_FLOAT16)

        try:
            if self._gpu_resources is None:
                self._gpu_resources = faiss.StandardGpuResources()

            self.content_index = self._cpu_to_gpu_index(
                faiss,
                self.content_index,
                self._gpu_resources,
                device,
                cloner_options,
            )
            self.sectionid_index = self._cpu_to_gpu_index(
                faiss,
                self.sectionid_index,
                self._gpu_resources,
                device,
                cloner_options,
            )
            self._faiss_gpu_enabled = True
            precision = "float16" if cfg.FAISS_GPU_USE_FLOAT16 else "float32"
            print(f"FAISS indices moved to GPU:{device} ({precision}) [{where}].")
        except Exception as exc:
            if not self._warned_no_gpu_faiss:
                print(f"FAISS GPU init failed; using CPU indices. Reason: {exc}")
                self._warned_no_gpu_faiss = True
            self._faiss_gpu_enabled = False

    @staticmethod
    def _cpu_to_gpu_index(faiss, index, gpu_resources, device: int, cloner_options):
        if cloner_options is None:
            return faiss.index_cpu_to_gpu(gpu_resources, device, index)

        try:
            return faiss.index_cpu_to_gpu(
                gpu_resources,
                device,
                index,
                cloner_options,
            )
        except TypeError:
            return faiss.index_cpu_to_gpu(gpu_resources, device, index)

    @staticmethod
    def _gpu_to_cpu_index(faiss, index):
        if hasattr(faiss, "index_gpu_to_cpu"):
            try:
                return faiss.index_gpu_to_cpu(index)
            except Exception:
                return index
        return index

    def _encode_query(self, text: str) -> np.ndarray:
        cache_size = max(0, int(cfg.VECTOR_QUERY_EMBED_CACHE_SIZE))
        if cache_size > 0:
            cached = self._query_embedding_cache.get(text)
            if cached is not None:
                self._query_embedding_cache.move_to_end(text)
                return cached

        vec = self.model.encode([text], normalize_embeddings=True).astype(np.float32)

        if cache_size > 0:
            self._query_embedding_cache[text] = vec
            self._query_embedding_cache.move_to_end(text)
            while len(self._query_embedding_cache) > cache_size:
                self._query_embedding_cache.popitem(last=False)

        return vec

    def _cache_get_result(
        self,
        key: tuple[str, int, bool],
    ) -> list[tuple[int, float]] | None:
        cache_size = max(0, int(cfg.VECTOR_QUERY_RESULT_CACHE_SIZE))
        if cache_size <= 0:
            return None

        cached = self._query_result_cache.get(key)
        if cached is None:
            return None

        self._query_result_cache.move_to_end(key)
        return list(cached)

    def _cache_set_result(
        self,
        key: tuple[str, int, bool],
        ranked: list[tuple[int, float]],
    ) -> None:
        cache_size = max(0, int(cfg.VECTOR_QUERY_RESULT_CACHE_SIZE))
        if cache_size <= 0:
            return

        self._query_result_cache[key] = list(ranked)
        self._query_result_cache.move_to_end(key)
        while len(self._query_result_cache) > cache_size:
            self._query_result_cache.popitem(last=False)

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
