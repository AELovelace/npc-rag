"""The wiki index: chunks + embedding matrix + BM25, searched with reciprocal rank fusion."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .wiki import Chunk, Page, chunk_pages

RRF_K = 60  # Standard reciprocal-rank-fusion constant: dampens the gap between rank 1 and rank 2.


@dataclass
class Hit:
    chunk: Chunk
    dense: float   # Cosine similarity between the question and the chunk (0..1 in practice).
    keyword: float # BM25 score (unbounded; 0 means no shared words).
    fused: float   # RRF score used for the final order.

    def to_dict(self) -> dict:
        d = self.chunk.to_dict()
        d.update(dense=round(self.dense, 4), keyword=round(self.keyword, 4), fused=round(self.fused, 5))
        return d


class WikiIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, meta: dict, embedder):
        self.chunks = chunks
        self.vectors = vectors                                  # One unit-length row per chunk.
        self.meta = meta                                        # Model, source and build time.
        self.embedder = embedder
        self.bm25 = BM25([c.embed_text() for c in chunks])      # Keyword side is rebuilt on load; it's instant.

    # ── Build / persist ──────────────────────────────────────────────────

    @classmethod
    def build(cls, pages: list[Page], embedder, public_base: str, max_words: int, source: str) -> "WikiIndex":
        chunks = chunk_pages(pages, public_base, max_words)
        if not chunks:
            raise ValueError("The wiki produced no chunks; refusing to build an empty index")
        vectors = embedder.passages([c.embed_text() for c in chunks])
        meta = {
            "model": embedder.name,
            "source": source,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pages": len(pages),
            "chunks": len(chunks),
            "dims": int(vectors.shape[1]),
        }
        return cls(chunks, vectors, meta, embedder)

    def save(self, directory: Path) -> None:
        """Write the three index files; each is written to a temp name first so a crash never leaves half a file."""
        directory.mkdir(parents=True, exist_ok=True)
        tmp_vec = directory / "vectors.tmp.npy"
        np.save(tmp_vec, self.vectors)
        os.replace(tmp_vec, directory / "vectors.npy")
        for name, payload in (("chunks.json", [c.to_dict() for c in self.chunks]), ("meta.json", self.meta)):
            tmp = directory / (name + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, directory / name)

    @classmethod
    def load(cls, directory: Path, embedder) -> "WikiIndex":
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta.get("model") != embedder.name:
            raise ValueError(f"Index was built with {meta.get('model')!r} but the service uses {embedder.name!r}; rebuild it")
        chunks = [Chunk(**c) for c in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
        vectors = np.load(directory / "vectors.npy")
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks.json and vectors.npy disagree; rebuild the index")
        return cls(chunks, vectors, meta, embedder)

    # ── Search ───────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 4) -> tuple[list[Hit], float]:
        """Top k chunks by fused dense + keyword rank, plus the best dense score over the whole wiki."""
        qv = self.embedder.queries([query])[0]
        dense = self.vectors @ qv                                   # Cosine similarity with every chunk.
        keyword = np.array(self.bm25.scores(query), dtype=np.float32)

        fused = np.zeros(len(self.chunks), dtype=np.float64)
        for rank, i in enumerate(np.argsort(-dense)):
            fused[i] += 1.0 / (RRF_K + rank + 1)                    # Every chunk gets its dense-rank share.
        for rank, i in enumerate(np.argsort(-keyword)):
            if keyword[i] <= 0:
                break                                               # Chunks sharing no words get no keyword share.
            fused[i] += 1.0 / (RRF_K + rank + 1)

        order = np.argsort(-fused)[:k]
        hits = [Hit(self.chunks[i], float(dense[i]), float(keyword[i]), float(fused[i])) for i in order]
        return hits, float(dense.max())  # Best dense = how strongly the wiki talks about this question at all.
