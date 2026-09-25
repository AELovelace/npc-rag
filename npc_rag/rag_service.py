"""RAG service (port 9092): searches the wiki index and hands back the best excerpts."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .common import api_key_guard
from .config import Settings
from .index import WikiIndex
from .wiki import load_pages

log = logging.getLogger("npc_rag.rag")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=12)


class ReindexRequest(BaseModel):
    source: str | None = None  # Override WIKI_SOURCE for this rebuild only (URL or local web/wiki folder).


def build_index(settings: Settings, embedder, source: str | None = None) -> WikiIndex:
    """Fetch the wiki, chunk + embed it and save it to INDEX_DIR."""
    src = source or settings.wiki_source
    pages = load_pages(src)
    index = WikiIndex.build(pages, embedder, settings.wiki_public_url, settings.chunk_words, src)
    index.save(settings.index_dir)
    log.info("Indexed %s pages into %s chunks from %s", index.meta["pages"], index.meta["chunks"], src)
    return index


def load_or_build(settings: Settings, embedder) -> WikiIndex:
    """Use the saved index; build it on first start or when the embedding model changed."""
    try:
        return WikiIndex.load(settings.index_dir, embedder)
    except (FileNotFoundError, ValueError) as exc:
        log.warning("No usable index (%s); building one from %s", exc, settings.wiki_source)
        return build_index(settings, embedder)


def create_app(settings: Settings, embedder=None, index: WikiIndex | None = None) -> FastAPI:
    """App factory; tests pass a stub embedder and a ready index."""
    state: dict = {"index": index, "embedder": embedder}
    rebuild_lock = asyncio.Lock()  # One rebuild at a time; searches keep using the old index meanwhile.

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["embedder"] is None:
            from .embedder import Embedder
            state["embedder"] = Embedder(settings.embed_model, settings.embed_cache, settings.embed_threads)
        if state["index"] is None:
            state["index"] = await asyncio.to_thread(load_or_build, settings, state["embedder"])
        yield

    app = FastAPI(title="npc-rag retrieval", lifespan=lifespan)
    guard = Depends(api_key_guard(settings.api_key))

    @app.get("/health")
    async def health():
        idx: WikiIndex = state["index"]
        return {"ok": idx is not None, "service": "rag", "index": idx.meta if idx else None}

    @app.post("/v1/search", dependencies=[guard])
    async def search(req: SearchRequest):
        idx: WikiIndex = state["index"]
        hits, best = await asyncio.to_thread(idx.search, req.query, req.k)  # Embedding is CPU work: keep the loop free.
        return {
            "query": req.query,
            "best_dense": round(best, 4),
            "relevant": best >= settings.min_relevance,                    # Does the wiki really talk about this?
            "results": [h.to_dict() for h in hits],
        }

    @app.post("/v1/reindex", dependencies=[guard])
    async def reindex(req: ReindexRequest | None = None):
        if rebuild_lock.locked():
            raise HTTPException(409, "a rebuild is already running")
        async with rebuild_lock:
            try:
                fresh = await asyncio.to_thread(build_index, settings, state["embedder"], req.source if req else None)
            except Exception as exc:  # Keep serving the old index if the wiki can't be fetched.
                log.exception("Reindex failed")
                raise HTTPException(502, f"reindex failed, old index kept: {exc}") from exc
            state["index"] = fresh                                         # Atomic swap for new searches.
        return {"ok": True, "index": fresh.meta}

    return app
