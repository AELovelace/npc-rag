"""CPU text embeddings through fastembed (ONNX runtime, no torch or GPU needed)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .bm25 import tokenize


def _normalize(m: np.ndarray) -> np.ndarray:
    """Unit-length rows so a dot product is cosine similarity."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


class Embedder:
    """fastembed wrapper: passages for documents and seed examples, queries for retrieval questions."""

    def __init__(self, model: str, cache_dir: Path, threads: int = 0):
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows without Developer Mode can't symlink; that's fine.
        from fastembed import TextEmbedding  # Imported here so tests with the stub never load onnxruntime.

        self.name = model
        self._model = TextEmbedding(model, cache_dir=str(cache_dir), threads=threads or None)  # Downloads once, then loads from cache.

    def passages(self, texts: list[str]) -> np.ndarray:
        """Embed documents (wiki chunks) or symmetric short texts (classifier examples)."""
        return _normalize(np.array(list(self._model.passage_embed(texts)), dtype=np.float32))

    def queries(self, texts: list[str]) -> np.ndarray:
        """Embed search queries; bge models add their retrieval instruction here."""
        return _normalize(np.array(list(self._model.query_embed(texts)), dtype=np.float32))


class HashEmbedder:
    """Deterministic bag-of-words vectors for tests: no download, same interface as Embedder."""

    name = "hash-test"

    def __init__(self, dims: int = 256):
        self.dims = dims

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dims, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dims] += 1.0  # Each word lights one bucket.
        return v

    def passages(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array([self._vec(t) for t in texts]).reshape(len(texts), self.dims))

    queries = passages  # The stub has no query instruction.
