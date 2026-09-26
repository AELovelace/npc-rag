"""npc-rag service (port 9092): wiki retrieval plus guarded NPC replies.

POST /v1/npc/chat is the front door: player message in, NPC reply out.
  1. classify  - Sakura's small model (9091) says game or chat; the wiki score decides if it's down
  2. retrieve  - game questions get the best-matching wiki sections from the local index
  3. generate  - Sakura's main model (9090) writes the reply from the persona + those sections
  4. guard     - replies that invent names or recite the rules are retried, then replaced (guard.py)
POST /v1/search and /v1/reindex expose retrieval on its own.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .classifier import CLASSIFY_PROMPT, GAME, classify, parse_verdict
from .common import Sessions, api_key_guard
from .config import Settings
from .guard import find_unsupported_terms, leaks_persona, retry_rule, select_hits, to_ascii
from .index import WikiIndex
from .llm import LlamaClient, LLMError
from .prompts import (LEAK_REPLY, build_messages, clean_name, fallback_reply, rules_text, tidy_reply,
                      unsure_reply)
from .wiki import load_pages

log = logging.getLogger("npc_rag")

FOLLOW_UP_WORDS = 5       # Shorter game messages ("and in Arcadia?") are searched together with the previous question.
RETRY_TEMPERATURE = 0.1   # A guarded retry runs cold so the model sticks to the notes.


class Turn(BaseModel):
    role: Literal["user", "assistant", "player", "npc"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    player_id: str | None = Field(default=None, max_length=128)    # Enables per-player memory when set.
    player_name: str | None = Field(default=None, max_length=64)
    npc_name: str | None = Field(default=None, max_length=32)       # Overrides NPC_NAME for this call.
    history: list[Turn] | None = Field(default=None, max_length=20) # Caller-managed memory; replaces the built-in one.


class ClassifyRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=12)


class ReindexRequest(BaseModel):
    source: str | None = None  # Override WIKI_SOURCE for this rebuild only (URL or local web/wiki folder).


def build_index(settings: Settings, embedder, source: str | None = None) -> WikiIndex:
    """Fetch the wiki, chunk + embed it and save it to INDEX_DIR."""
    src = source or settings.wiki_source
    index = WikiIndex.build(load_pages(src), embedder, settings.wiki_public_url, settings.chunk_words, src)
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


def _normalise_history(turns: list[Turn], limit: int) -> list[dict]:
    """Caller history in OpenAI roles, newest `limit` messages only."""
    roles = {"player": "user", "npc": "assistant"}
    return [{"role": roles.get(t.role, t.role), "content": t.content} for t in turns][-limit:]


def create_app(settings: Settings, *, embedder=None, index: WikiIndex | None = None,
               llm: LlamaClient | None = None, classifier_llm: LlamaClient | None = None) -> FastAPI:
    """App factory; tests inject a stub embedder, a ready index and mocked model servers."""
    state: dict = {"index": index, "embedder": embedder}
    sessions = Sessions(settings.session_turns, settings.session_ttl)
    in_flight: set[str] = set()     # Players with a reply being generated; a second message waits its turn.
    rebuild_lock = asyncio.Lock()   # One rebuild at a time; searches keep using the old index meanwhile.

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["embedder"] is None:
            from .embedder import Embedder
            state["embedder"] = Embedder(settings.embed_model, settings.embed_cache, settings.embed_threads)
        if state["index"] is None:
            state["index"] = await asyncio.to_thread(load_or_build, settings, state["embedder"])
        state["llm"] = llm or LlamaClient(settings.llm_url, settings.llm_model, settings.llm_api_key,
                                          settings.llm_concurrency)
        state["classifier"] = classifier_llm or LlamaClient(settings.classifier_url, settings.classifier_model,
                                                            settings.llm_api_key, settings.llm_concurrency)
        state["persona"] = settings.persona_path.read_text(encoding="utf-8")
        yield
        await state["llm"].aclose()
        await state["classifier"].aclose()

    app = FastAPI(title="npc-rag", lifespan=lifespan)
    guard = Depends(api_key_guard(settings.api_key))

    async def search(query: str, k: int) -> dict:
        hits, best = await asyncio.to_thread(state["index"].search, query, k)  # Embedding is CPU work: keep the loop free.
        return {"query": query, "best_dense": round(best, 4), "relevant": best >= settings.min_relevance,
                "results": [h.to_dict() for h in hits]}

    async def ask_classifier(message: str) -> str | None:
        try:
            text = await state["classifier"].chat(
                [{"role": "system", "content": CLASSIFY_PROMPT}, {"role": "user", "content": message}],
                max_tokens=4, temperature=0.0, timeout=settings.classifier_timeout)
            return parse_verdict(text)
        except LLMError as exc:
            log.warning("Classifier unavailable, using the wiki score: %s", exc)
            return None

    async def classify_with_search(text: str, cache: dict):
        """Classify; the wiki fallback's search result is kept so a game answer can reuse it."""
        async def evidence(q: str) -> float | None:
            cache["search"] = await search(q, settings.top_k)
            return cache["search"]["best_dense"]
        return await classify(text, ask_llm=ask_classifier, evidence=evidence, wiki_evidence=settings.rag_game_evidence)

    @app.get("/health")
    async def health():
        llm_ok, clf_ok = await asyncio.gather(state["llm"].health(), state["classifier"].health())
        idx: WikiIndex = state["index"]
        return {"ok": idx is not None, "service": "npc-rag", "llm": llm_ok, "classifier": clf_ok,
                "llm_url": settings.llm_url, "classifier_url": settings.classifier_url,
                "index": idx.meta if idx else None}

    @app.post("/v1/search", dependencies=[guard])
    async def search_endpoint(req: SearchRequest):
        return await search(req.query, req.k)

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

    @app.post("/v1/classify", dependencies=[guard])
    async def classify_endpoint(req: ClassifyRequest):
        return (await classify_with_search(req.message.strip(), {})).to_dict()

    @app.delete("/v1/npc/session/{player_id}", dependencies=[guard])
    async def forget(player_id: str):
        sessions.clear(player_id)
        return {"ok": True}

    @app.post("/v1/npc/chat", dependencies=[guard])
    async def chat(req: ChatRequest):
        message = " ".join(req.message.split())  # One line, no stray whitespace.
        if not message or len(message) > settings.max_message_chars:
            raise HTTPException(422, f"message must be 1-{settings.max_message_chars} characters")
        pid = req.player_id
        if pid and pid in in_flight:
            raise HTTPException(429, "this player already has a reply on the way")
        if pid:
            in_flight.add(pid)
        try:
            return await _chat(req, message)
        finally:
            if pid:
                in_flight.discard(pid)

    async def generate(messages: list[dict], persona: str, sources: list[str]) -> tuple[str | None, str]:
        """Ask the model, check the reply, retry once if it names unsupported things.

        Returns (raw reply or None, guard verdict). None means the reply was rejected.
        """
        attempt_messages = messages
        for attempt in range(settings.guard_retries + 1):
            raw = await state["llm"].chat(attempt_messages, max_tokens=settings.max_tokens,
                                          temperature=settings.temperature if attempt == 0 else RETRY_TEMPERATURE,
                                          timeout=settings.llm_timeout)
            if leaks_persona(raw, persona):
                return LEAK_REPLY, "leak_blocked"
            terms = find_unsupported_terms(raw, sources)
            if not terms:
                return raw, "passed" if attempt == 0 else "passed_after_retry"
            log.info("Guard: reply named %s; %s", terms, "retrying" if attempt < settings.guard_retries else "giving up")
            system = {**messages[0], "content": messages[0]["content"] + retry_rule(terms)}
            attempt_messages = [system, *messages[1:]]
        return None, "unsupported_terms"

    async def _chat(req: ChatRequest, message: str) -> dict:
        t0 = time.perf_counter()
        timings: dict[str, int] = {}
        npc_name = clean_name(req.npc_name, settings.npc_name)
        player_name = clean_name(req.player_name, "traveller")
        history = (_normalise_history(req.history, settings.session_turns) if req.history is not None
                   else sessions.get(req.player_id) if req.player_id else [])

        # 1. Classify (the wiki fallback may already search; keep that result).
        cache: dict = {}
        verdict = await classify_with_search(message, cache)
        timings["classify"] = int((time.perf_counter() - t0) * 1000)

        # 2. Retrieve for game questions. Short follow-ups borrow the previous question's words.
        hits: list[dict] = []
        if verdict.category == GAME:
            t1 = time.perf_counter()
            last_user = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
            query = f"{last_user} {message}" if last_user and len(message.split()) < FOLLOW_UP_WORDS else message
            result = cache.get("search") if query == message else None
            hits = (result or await search(query, settings.top_k))["results"]
            timings["retrieve"] = int((time.perf_counter() - t1) * 1000)
        notes = select_hits(hits, settings.min_relevance, settings.guard_keep_margin, settings.top_k,
                            settings.guard_min_words)

        # 3 + 4. Generate and guard. A game question with no relevant notes never reaches the model.
        t2 = time.perf_counter()
        fallback = False
        if verdict.category == GAME and not notes:
            reply, guard_verdict = unsure_reply(verdict.category, hits), "no_relevant_notes"
        else:
            messages = build_messages(npc_name=npc_name, player_name=player_name, persona=state["persona"],
                                      category=verdict.category, message=message, history=history, hits=notes)
            persona = rules_text(npc_name=npc_name, player_name=player_name, persona=state["persona"])
            sources = [m["content"] for m in messages] + [npc_name, player_name, *settings.guard_allowed_terms]
            try:
                raw, guard_verdict = await generate(messages, persona, sources)
                reply = (tidy_reply(raw, npc_name, settings.max_reply_chars) if raw is not None
                         else unsure_reply(verdict.category, hits))
            except LLMError as exc:
                log.warning("Using fallback line: %s", exc)
                reply, fallback, guard_verdict = fallback_reply(verdict.category, hits), True, "llm_unavailable"
            if not reply:
                reply, fallback = fallback_reply(verdict.category, hits), True
        reply = to_ascii(reply)  # GameMaker fonts often lack curly quotes and dashes.
        timings["generate"] = int((time.perf_counter() - t2) * 1000)
        timings["total"] = int((time.perf_counter() - t0) * 1000)

        if req.player_id and req.history is None and not fallback:
            sessions.add(req.player_id, message, reply)  # Fallback lines aren't worth remembering.

        sources_out, seen = [], set()
        for h in notes if guard_verdict.startswith("passed") else []:  # Only link what the reply was written from.
            if h["url"] not in seen:
                seen.add(h["url"])
                sources_out.append({"title": h["page_title"], "section": h["heading"], "url": h["url"], "score": h["dense"]})

        log.info("%s %s/%s %s %dms: %r", req.player_id or "-", verdict.category, verdict.method, guard_verdict,
                 timings["total"], message[:80])
        return {
            "reply": reply,
            "npc_name": npc_name,
            "category": verdict.category,
            "method": verdict.method,
            "relevant": bool(notes),
            "fallback": fallback,
            "guard": guard_verdict,
            "sources": sources_out[:3],
            "timings_ms": timings,
        }

    return app
