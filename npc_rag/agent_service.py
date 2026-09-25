"""Classifier + agent service (port 9091).

POST /v1/npc/chat is the front door: player message in, NPC reply out.
  1. classify   - game question or general chat (classifier.py, CPU)
  2. retrieve   - game questions get wiki excerpts from the RAG service (9092)
  3. generate   - llama.cpp (9090) writes the NPC's reply from the persona + excerpts
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .classifier import GAME, TIEBREAK_PROMPT, Classifier, load_seeds, parse_tiebreak
from .common import Sessions, api_key_guard
from .config import Settings
from .llm import LlamaClient, LLMError
from .prompts import build_messages, clean_name, fallback_reply, tidy_reply

log = logging.getLogger("npc_rag.agent")

FOLLOW_UP_WORDS = 5  # Shorter game messages ("and in Arcadia?") are searched together with the previous question.


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


class RagClient:
    """Talks to the RAG service; returns None instead of raising so chat can degrade gracefully."""

    def __init__(self, base_url: str, api_key: str, transport: httpx.AsyncBaseTransport | None = None):
        headers = {"X-Api-Key": api_key} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, transport=transport, timeout=10)

    async def search(self, query: str, k: int) -> dict | None:
        try:
            r = await self._client.post("/v1/search", json={"query": query, "k": k})
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("RAG search failed: %r", exc)
            return None

    async def health(self) -> bool:
        try:
            return (await self._client.get("/health", timeout=3)).status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


def _normalise_history(turns: list[Turn], limit: int) -> list[dict]:
    """Caller history in OpenAI roles, newest `limit` messages only."""
    roles = {"player": "user", "npc": "assistant"}
    return [{"role": roles.get(t.role, t.role), "content": t.content} for t in turns][-limit:]


def create_app(settings: Settings, *, embedder=None, llm: LlamaClient | None = None,
               rag_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """App factory; tests inject a stub embedder, a mocked llama.cpp and a mocked RAG service."""
    state: dict = {}
    sessions = Sessions(settings.session_turns, settings.session_ttl)
    in_flight: set[str] = set()  # Players with a reply being generated; a second message waits its turn.

    async def tiebreak(message: str) -> str | None:
        try:
            text = await state["llm"].chat(
                [{"role": "system", "content": TIEBREAK_PROMPT}, {"role": "user", "content": message}],
                max_tokens=4, temperature=0.0, timeout=10)
            return parse_tiebreak(text)
        except LLMError as exc:
            log.warning("Tiebreak skipped: %s", exc)
            return None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        emb = embedder
        if emb is None:
            from .embedder import Embedder
            emb = Embedder(settings.embed_model, settings.embed_cache, settings.embed_threads)
        game, general = load_seeds(settings.seeds_path)
        state["llm"] = llm or LlamaClient(settings.llm_url, settings.llm_model, settings.llm_api_key, settings.llm_concurrency)
        state["rag"] = RagClient(settings.rag_url, settings.api_key, rag_transport)
        state["classifier"] = Classifier(
            emb, game, general, margin=settings.classifier_margin, wiki_evidence=settings.rag_game_evidence,
            tiebreak=tiebreak if settings.llm_tiebreak else None)
        state["persona"] = settings.persona_path.read_text(encoding="utf-8")
        yield
        await state["rag"].aclose()
        await state["llm"].aclose()

    app = FastAPI(title="npc-rag classifier + agent", lifespan=lifespan)
    guard = Depends(api_key_guard(settings.api_key))

    async def classify_with_search(text: str, cache: dict):
        """Classify; the wiki evidence step's search result is kept so a game answer can reuse it."""
        async def evidence(q: str) -> float | None:
            cache["search"] = await state["rag"].search(q, settings.top_k)
            return cache["search"]["best_dense"] if cache["search"] else None
        return await state["classifier"].classify(text, evidence=evidence)

    @app.get("/health")
    async def health():
        llm_ok, rag_ok = await asyncio.gather(state["llm"].health(), state["rag"].health())
        return {"ok": True, "service": "agent", "llm": llm_ok, "rag": rag_ok, "llm_url": settings.llm_url}

    @app.post("/v1/classify", dependencies=[guard])
    async def classify(req: ClassifyRequest):
        verdict = await classify_with_search(req.message.strip(), {})
        return verdict.to_dict()

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

    async def _chat(req: ChatRequest, message: str) -> dict:
        t0 = time.perf_counter()
        timings: dict[str, int] = {}
        npc_name = clean_name(req.npc_name, settings.npc_name)
        player_name = clean_name(req.player_name, "traveller")
        history = (_normalise_history(req.history, settings.session_turns) if req.history is not None
                   else sessions.get(req.player_id) if req.player_id else [])

        # 1. Classify (the evidence step may already search the wiki; keep that result).
        cache: dict = {}
        verdict = await classify_with_search(message, cache)
        timings["classify"] = int((time.perf_counter() - t0) * 1000)

        # 2. Retrieve for game questions. Short follow-ups borrow the previous question's words.
        hits: list[dict] = []
        relevant = False
        if verdict.category == GAME:
            t1 = time.perf_counter()
            last_user = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
            query = f"{last_user} {message}" if last_user and len(message.split()) < FOLLOW_UP_WORDS else message
            search = cache.get("search") if query == message else None
            search = search or await state["rag"].search(query, settings.top_k)
            if search:
                hits, relevant = search["results"], bool(search["relevant"])
            timings["retrieve"] = int((time.perf_counter() - t1) * 1000)

        # 3. Generate the NPC's line.
        t2 = time.perf_counter()
        messages = build_messages(npc_name=npc_name, player_name=player_name, persona=state["persona"],
                                  category=verdict.category, message=message, history=history,
                                  hits=hits, relevant=relevant)
        fallback = False
        try:
            raw = await state["llm"].chat(messages, max_tokens=settings.max_tokens,
                                          temperature=settings.temperature, timeout=settings.llm_timeout)
            reply = tidy_reply(raw, npc_name, settings.max_reply_chars)
        except LLMError as exc:
            log.warning("Using fallback line: %s", exc)
            reply, fallback = fallback_reply(verdict.category, hits), True
        if not reply:
            reply, fallback = fallback_reply(verdict.category, hits), True
        timings["generate"] = int((time.perf_counter() - t2) * 1000)
        timings["total"] = int((time.perf_counter() - t0) * 1000)

        if req.player_id and req.history is None and not fallback:
            sessions.add(req.player_id, message, reply)  # Fallback lines aren't worth remembering.

        sources, seen = [], set()
        for h in hits if relevant else []:
            if h["url"] not in seen:
                seen.add(h["url"])
                sources.append({"title": h["page_title"], "section": h["heading"], "url": h["url"], "score": h["dense"]})

        log.info("%s %s/%s %.2f %dms: %r", req.player_id or "-", verdict.category, verdict.method,
                 verdict.confidence, timings["total"], message[:80])
        return {
            "reply": reply,
            "npc_name": npc_name,
            "category": verdict.category,
            "confidence": round(verdict.confidence, 3),
            "method": verdict.method,
            "relevant": relevant,
            "fallback": fallback,
            "sources": sources[:3],
            "timings_ms": timings,
        }

    return app
