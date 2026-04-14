"""Vector retriever wrapping VectorIndex."""

from pathlib import Path
from src.indexing.vector_index import VectorIndex


class VectorRetriever:
    def __init__(self, index: VectorIndex):
        self.index = index

    @classmethod
    def load(cls, content_path: Path, sectionid_path: Path, meta_path: Path, model_name: str):
        index = VectorIndex(model_name)
        index.load(content_path, sectionid_path, meta_path)
        return cls(index)

    def query(self, text: str, k: int = 10) -> list[dict]:
        """Return top-k chunks by cosine similarity to `text`."""
        return self.index.query(text, k)
