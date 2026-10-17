"""npc-rag service (port 9092): wiki retrieval plus guarded NPC replies.

POST /v1/npc/chat is the front door: player message in, NPC reply out.
  1. classify  - strong wiki evidence decides immediately; otherwise the small model says game or chat
  2. retrieve  - game questions get the best-matching wiki sections from the local index
  3. generate  - Sakura's main model (9090) writes the reply from the persona + those sections
  4. guard     - replies that invent names or recite the rules are retried, then replaced (guard.py)
POST /v1/search and /v1/reindex expose retrieval on its own.
POST /v1/companion/chat and /v1/companion/lore serve the online companions (companion/router.py, a LangGraph pipeline).
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .speaking_style import SpeakingStyle, style_prompt
from .classifier import CLASSIFIER_REVISION, GAME, GENERAL, ModelDecisionCache, Verdict, classify
from .classifier_backend import make_classifier
from .common import Sessions, api_key_guard
from .config import Settings
from .guard import find_unsupported_terms, leaks_persona, retry_rule, select_hits, to_ascii
from .index import WikiIndex
from .llm import LlamaClient, LLMError, background_chat, make_reply_llm
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
    speaking_style: SpeakingStyle | None = None  # Select dialogue tone without modifying the authored persona.
    persona: str | None = Field(default=None, max_length=1500)      # Overrides data/persona.md for this call (companions speak as themselves).
    category: Literal["game", "general", "chat"] | None = None      # Caller already classified the message: skip the 9091 step.


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
               llm: LlamaClient | None = None, classifier_llm: LlamaClient | None = None, participation_backend=None) -> FastAPI:
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
        state["llm"] = llm or make_reply_llm(settings)  # The same guarded pipelines support either local or overflow generation.
        state["classifier"] = make_classifier(settings, client=classifier_llm)
        from .participation import make_participation
        try:
            state['participation'] = (participation_backend or make_participation(settings, classifier=state['classifier'])) if settings.participation_enabled else None
            await state["classifier"].start()  # Native model must load successfully before this service accepts traffic.
            if state['participation'] is not None:
                await state['participation'].start()
            state["classifier_cache"] = ModelDecisionCache(settings.classifier_cache_size, settings.classifier_cache_ttl)
            state["persona"] = settings.persona_path.read_text(encoding="utf-8")
            yield
        finally:
            if state.get('participation') is not None:
                await state['participation'].aclose()
            await state["classifier"].aclose()
            await state["llm"].aclose()

    app = FastAPI(title="npc-rag", lifespan=lifespan)
    guard = Depends(api_key_guard(settings.api_key))

    async def search(query: str, k: int) -> dict:
        hits, best, coverage = await asyncio.to_thread(state["index"].search_evidence, query, k)  # Embedding is CPU work: keep the loop free.
        return {"query": query, "best_dense": round(best, 4), "keyword_coverage": round(coverage, 4),
                "relevant": best >= settings.min_relevance,
                "results": [h.to_dict() for h in hits]}

    classifier_identity, identity_checked = None, float("-inf")
    identity_lock = asyncio.Lock()

    async def identity_known() -> bool:
        """Recheck advertised model identity at most every 30 seconds; a failed check disables caching."""
        nonlocal classifier_identity, identity_checked
        async with identity_lock:
            if time.monotonic() - identity_checked >= 30:
                identity = await state["classifier"].model_identity()
                if identity is None or identity != classifier_identity:
                    state["classifier_cache"].clear()  # A hot swap or an unknown backend must not reuse old labels.
                classifier_identity, identity_checked = identity, time.monotonic()
            return classifier_identity is not None

    async def ask_classifier_uncached(message: str) -> str | None:
        try:
            return await asyncio.wait_for(state["classifier"].decide(message),
                timeout=settings.classifier_timeout)  # Bound queue + HTTP/native wait; native work may finish afterward.
        except asyncio.TimeoutError:
            log.warning("Classifier deadline exceeded (queue + request), using the wiki score")
            return None
        except LLMError as exc:
            log.warning("Classifier unavailable, using the wiki score: %s", exc)
            return None

    async def ask_classifier(message: str) -> str | None:
        async def cached():
            if not await identity_known():
                return await ask_classifier_uncached(message)  # Keep serving when optional identity discovery is unavailable.
            return await state["classifier_cache"].get(message, lambda: ask_classifier_uncached(message))
        try:
            return await asyncio.wait_for(cached(), timeout=settings.classifier_timeout)  # Identity lookup and shared waits count too.
        except asyncio.TimeoutError:
            return None

    async def classify_with_search(text: str, cache: dict):
        """Classify; the wiki fallback's search result is kept so a game answer can reuse it."""
        async def evidence(q: str) -> float | None:
            cache["search"] = await search(q, settings.top_k)
            return {"dense": cache["search"]["best_dense"], "coverage": cache["search"]["keyword_coverage"]}
        return await classify(text, ask_llm=ask_classifier, evidence=evidence, wiki_evidence=settings.rag_game_evidence,
                              keyword_coverage=settings.rag_keyword_coverage, keyword_dense=settings.rag_keyword_dense,
                              wiki_first=settings.classifier_backend == "decider")  # Native runs can't be abandoned once started.

    @app.get("/health")
    async def health():
        llm_ok, clf_ok = await asyncio.gather(state["llm"].health(), state["classifier"].health())
        idx: WikiIndex = state["index"]
        return {"ok": idx is not None, "service": "npc-rag", "llm": llm_ok, "classifier": clf_ok,
                "llm_url": settings.llm_url,
                "classifier_url": settings.classifier_url if settings.classifier_backend == "llamacpp" else None,
                "overflow": state["llm"].view() if hasattr(state["llm"], "view") else {"enabled": False},
                "classification": {"policy": CLASSIFIER_REVISION, "cache": state["classifier_cache"].view(),
                                   **state["classifier"].view()},
                'participation': ({'enabled': True, 'healthy': await state['participation'].health(),
                                   **state['participation'].view()} if state['participation'] else {'enabled': False}),
                "index": idx.meta if idx else None}

    @app.post('/v1/participation/decide', dependencies=[guard])
    async def participation_endpoint(context: dict):
        from .participation_policy import validate_state
        backend = state.get('participation')
        if backend is None:
            raise HTTPException(503, 'participation model is disabled')
        try:
            validated = validate_state(context)
        except (ValueError, TypeError, KeyError):
            raise HTTPException(422, 'invalid participation context') from None
        began = time.perf_counter()
        try:
            decision = await asyncio.wait_for(backend.decide(validated), settings.participation_timeout)
        except (asyncio.TimeoutError, LLMError):
            raise HTTPException(503, 'participation model unavailable or busy') from None
        metadata = backend.view()
        return {'decision': decision.label, 'join': decision.label == 'join', 'join_probability': decision.join_probability,
                'policy': metadata['decision_policy'], 'model': metadata.get('release') or metadata.get('model') or settings.participation_model,
                'timings_ms': {'participation': round((time.perf_counter() - began) * 1000, 2)}}

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

    async def generate(messages: list[dict], persona: str, sources: list[str], max_tokens: int | None = None,
                       *, lane: str = "local") -> tuple[str | None, str]:
        """Ask the model, check the reply, retry once if it names unsupported things.

        lane: "paid" may overflow to OpenAI (player conversations only), "local" waits for a local slot,
        "background" runs only while local slots are free for players (ambient NPC life). Anything else is "local".
        Returns (raw reply or None, guard verdict). None means the reply was rejected.
        """
        attempt_messages = messages
        llm = state["llm"]
        if lane == "paid":
            model = getattr(llm, "reply", llm.chat)  # The only path that can bill: OverflowClient.reply().
        elif lane == "background":
            model = lambda m, **kw: background_chat(llm, m, reserve=settings.ambient_local_reserve, **kw)  # Never the cloud.
        else:
            model = llm.chat  # Local queue; OverflowClient.chat() never overflows.
        for attempt in range(settings.guard_retries + 1):
            raw = await model(attempt_messages, max_tokens=max_tokens or settings.max_tokens,
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

    from .companion.router import companion_router  # Companions share the classifier, wiki search and guarded generate.
    app.include_router(companion_router(settings, state, search=search, classify_with_search=classify_with_search,
                                        generate=generate))

    async def _chat(req: ChatRequest, message: str) -> dict:
        t0 = time.perf_counter()
        timings: dict[str, int] = {}
        npc_name = clean_name(req.npc_name, settings.npc_name)
        player_name = clean_name(req.player_name, "traveller")
        history = (_normalise_history(req.history, settings.session_turns) if req.history is not None
                   else sessions.get(req.player_id) if req.player_id else [])

        persona_text = ((req.persona or "").strip() or state["persona"]) + style_prompt(req.speaking_style)  # A caller persona replaces Pip's for this reply only.

        # 1. Classify (the wiki fallback may already search; keep that result). A caller verdict skips the model.
        cache: dict = {}
        if req.category:
            verdict = Verdict(GAME if req.category == "game" else GENERAL, "caller")  # "chat" is the game server's word for general.
        else:
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
            messages = build_messages(npc_name=npc_name, player_name=player_name, persona=persona_text,
                                      category=verdict.category, message=message, history=history, hits=notes)
            persona = rules_text(npc_name=npc_name, player_name=player_name, persona=persona_text)  # The leak guard checks the persona actually used.
            sources = [m["content"] for m in messages] + [npc_name, player_name, *settings.guard_allowed_terms]
            try:
                raw, guard_verdict = await generate(messages, persona, sources, lane="paid")  # Pip answers players: may overflow.
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

        log.info("%s %s/%s %s %dms [%s]: %r", req.player_id or "-", verdict.category, verdict.method, guard_verdict,
                 timings["total"], verdict.brief(), message[:80])  # Scores let reviewers find model/wiki disagreements to label.
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
