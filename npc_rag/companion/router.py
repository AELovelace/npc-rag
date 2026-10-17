"""Companion endpoints (mounted on the same 9092 app as Pip, same API key).

POST /v1/companion/lore   the game server pushes every companion's backstory (on boot, after edits, and on a 409)
GET  /v1/companion/lore   which lore version this node holds
POST /v1/companion/chat   one companion reply through the LangGraph pipeline (graph.py); ambient_* / shop_beckon kinds are
                          resident NPC lines, local-only: 429 "local_busy" when local slots are kept for players
POST /v1/companion/impressions   0-3 things the player said about themselves, after a chat (reflect.py)
POST /v1/companion/summarize     fold old memories into the relationship summary (reflect.py)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..speaking_style import SpeakingStyle
from ..common import api_key_guard
from ..llm import LLMError, LocalBusy, track_lanes
from ..prompts import clean_name
from .graph import build_companion_graph
from .prompts import AMBIENT_KINDS, PAID_KINDS
from .lore import EmbedCache, LoreSection, LoreStore
from .reflect import impressions_messages, parse_impressions, summary_messages, tidy_summary, impressions_grammar

log = logging.getLogger("npc_rag.companion")


class Turn(BaseModel):
    role: Literal["user", "assistant", "player", "npc"]
    content: str = Field(max_length=2000)


class MemoryIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=400)
    importance: int = Field(default=2, ge=1, le=5)
    age_days: float = Field(default=0.0, ge=0)
    embarrassing: bool = False


class GossipIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=400)          # Already anonymous: written from templates on the game server.


class PartnerIn(BaseModel):
    name: str = Field(min_length=1, max_length=32)
    persona: str = Field(default="", max_length=600)
    line: str = Field(default="", max_length=300)               # What the partner just said (empty: start the exchange).


class ThreadLine(BaseModel):
    speaker: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=300)


class CompanionChatRequest(BaseModel):
    message: str = Field(min_length=1)
    npc_id: str = Field(min_length=1, max_length=64)          # followers-data.json key; selects the lore.
    npc_name: str = Field(min_length=1, max_length=32)
    player_name: str | None = Field(default=None, max_length=64)
    persona: str = Field(default="", max_length=1500)
    speaking_style: SpeakingStyle | None = None  # Optional selector; voice remains example dialogue.
    voice: list[str] = Field(default_factory=list, max_length=5)  # Example lines; imitated, never leak-checked.
    history: list[Turn] = Field(default_factory=list, max_length=20)
    category: Literal["game", "general", "chat"] | None = None
    bond: int = Field(default=0, ge=0, le=100)
    flags: list[str] = Field(default_factory=list, max_length=200)
    lore_version: str | None = Field(default=None, max_length=64)
    memories: list[MemoryIn] = Field(default_factory=list, max_length=80)
    summary: str = Field(default="", max_length=1200)
    gossip: list[GossipIn] = Field(default_factory=list, max_length=5)
    audience: int = Field(default=0, ge=0, le=500)          # Other players in earshot, not counting the speaker.
    slip: MemoryIn | None = None                              # Embarrassing memory the game server rolled to let slip.
    kind: Literal["chat", "emote", "reaction", "musing", "banter", "ambient_open", "ambient_reply", "ambient_greet",
                  "ambient_chime", "ambient_answer", "shop_beckon"] = "chat"  # COMPANION_LIFELIKE_PLAN.md / AMBIENT_NPC_LIFE_PLAN.md.
    situation: list[Annotated[str, Field(max_length=160)]] = Field(default_factory=list, max_length=10)  # Short word facts ("badly hurt").
    about: str = Field(default="", max_length=300)            # reaction: what just happened.
    partner: PartnerIn | None = None                          # banter / ambient: the character being spoken to.
    thread: list[ThreadLine] = Field(default_factory=list, max_length=8)  # ambient: the conversation so far, oldest first.
    topic: str = Field(default="", max_length=160)            # ambient_open / ambient_greet: what to bring up this time (rotated by the game server).
    said: list[Annotated[str, Field(max_length=300)]] = Field(default_factory=list, max_length=12)  # ambient: this character's recent lines from earlier conversations.


class ImpressionsRequest(BaseModel):
    npc_name: str = Field(min_length=1, max_length=32)
    player_name: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=500)
    reply: str = Field(default="", max_length=600)


class SummarizeRequest(BaseModel):
    npc_name: str = Field(min_length=1, max_length=32)
    player_name: str = Field(min_length=1, max_length=64)
    summary: str = Field(default="", max_length=1200)
    memories: list[str] = Field(min_length=1, max_length=120)


class LoreSectionIn(BaseModel):
    npc: str = Field(min_length=1, max_length=64)
    id: str = Field(min_length=1, max_length=64)
    heading: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=1200)
    reveal_bond: int = Field(default=0, ge=0, le=100)
    reveal_flag: str = Field(default="", max_length=64)
    draft: bool = False


class LorePush(BaseModel):
    version: str = Field(min_length=1, max_length=64)
    sections: list[LoreSectionIn] = Field(default_factory=list, max_length=1000)


def _history(turns: list[Turn], limit: int) -> list[dict]:
    roles = {"player": "user", "npc": "assistant"}
    return [{"role": roles.get(t.role, t.role), "content": t.content} for t in turns][-limit:]


def companion_router(settings, state: dict, *, search, classify_with_search, generate) -> APIRouter:
    router = APIRouter(dependencies=[Depends(api_key_guard(settings.api_key))])
    state.setdefault("lore", None)
    lore_lock = asyncio.Lock()
    holder: dict = {}

    def graph():
        if "graph" not in holder:  # Built on first use: the embedder only exists once the app's lifespan has started.
            holder["cache"] = EmbedCache(state["embedder"], settings.companion_embed_cache)
            holder["graph"] = build_companion_graph(settings, state, search=search, classify_with_search=classify_with_search,
                                                    generate=generate, embed_cache=holder["cache"])
        return holder["graph"]

    @router.get("/v1/companion/lore")
    async def lore_status():
        store: LoreStore | None = state["lore"]
        return {"version": store.version if store else None, "sections": store.count() if store else 0}

    @router.post("/v1/companion/lore")
    async def push_lore(body: LorePush):
        async with lore_lock:
            store = await asyncio.to_thread(LoreStore.build, body.version,
                                            [LoreSection(**s.model_dump()) for s in body.sections], state["embedder"])
            state["lore"] = store                                # Atomic swap for new chats.
        log.info("Lore %s: %d sections for %d companions", body.version, store.count(), len(store.sections))
        return {"ok": True, "version": store.version, "sections": store.count()}

    @router.post("/v1/companion/chat")
    async def chat(req: CompanionChatRequest):
        message = " ".join(req.message.split())
        if not message or len(message) > settings.max_message_chars:
            raise HTTPException(422, f"message must be 1-{settings.max_message_chars} characters")
        store: LoreStore | None = state["lore"]
        if req.lore_version and (store is None or store.version != req.lore_version):
            raise HTTPException(409, "lore_version mismatch: push /v1/companion/lore first")  # The game server re-pushes and retries.
        t0 = time.perf_counter()
        npc_name, player_name = clean_name(req.npc_name, "Companion"), clean_name(req.player_name, "traveller")
        lanes = track_lanes()  # Records a paid call anywhere below; ambient kinds can never make one.
        try:
            out = await graph().ainvoke({"req": req, "message": message, "history": _history(req.history, settings.session_turns),
                                         "npc_name": npc_name, "player_name": player_name, "timings": {}})
        except LocalBusy:
            raise HTTPException(429, "local_busy") from None  # 4xx: the game server skips this line without marking the node down.
        except LLMError as exc:
            log.warning("Companion model unavailable: %s", exc)
            raise HTTPException(503, "the reply model is unavailable on this node") from exc  # Lets the game server's pool try the other node.
        timings = {**out.get("timings", {}), "total": int((time.perf_counter() - t0) * 1000)}
        lane = "cloud" if lanes.get("cloud") else "local"
        if lane == "cloud" and req.kind not in PAID_KINDS:
            log.error("COST GUARD: %s line for %s used the paid cloud", req.kind, req.npc_id)  # Must never happen; GM panel alarms too.
        lore_ids = [sec.id for sec, _ in out.get("lore", [])]
        log.info("companion %s %s/%s %s lore=%s locked=%s mem=%d gossip=%s slip=%s %dms [%s]: %r", req.npc_id, out["category"], out["method"],
                 out["guard"], lore_ids, bool(out.get("locked")), len(out.get("memories", [])), bool(out.get("gossip")), out.get("slipped"),
                 timings["total"], out.get("scores", "-"), message[:80])  # Scores make model/wiki disagreements reviewable.
        return {
            "reply": out["reply"],
            "npc_name": npc_name,
            "category": out["category"],
            "method": out["method"],
            "guard": out["guard"],
            "fallback": False,
            "lore": lore_ids,
            "locked_secret": bool(out.get("locked")),  # A gated section matched the message; only a marker was in the prompt.
            "memories_offered": [m["id"] for m in out.get("memories", [])],
            "gossip": out["gossip"]["id"] if out.get("gossip") else None,
            "slip": {"memory_id": out["slip"]["id"]} if out.get("slipped") else None,
            "sources": [{"title": h["page_title"], "section": h["heading"], "url": h["url"]}
                        for h in out.get("notes", [])][:3] if out["guard"].startswith("passed") else [],
            "timings_ms": timings,
            "lane": lane,
        }

    @router.post("/v1/companion/impressions")
    async def impressions(req: ImpressionsRequest):
        npc, player = clean_name(req.npc_name, "Companion"), clean_name(req.player_name, "traveller")
        try:
            raw = await state["llm"].chat(impressions_messages(npc=npc, player=player, message=" ".join(req.message.split()),
                                                               reply=" ".join(req.reply.split())),
                                          max_tokens=220, temperature=0.0, timeout=settings.llm_timeout,
                                          grammar=impressions_grammar(player))  # Valid JSON from any model (llama.cpp GBNF).
        except LLMError as exc:
            raise HTTPException(503, "the reply model is unavailable on this node") from exc  # The game server retries elsewhere later.
        return {"facts": parse_impressions(raw, player, message=req.message)}

    @router.post("/v1/companion/summarize")
    async def summarize(req: SummarizeRequest):
        npc, player = clean_name(req.npc_name, "Companion"), clean_name(req.player_name, "traveller")
        memories = [" ".join(m.split())[:400] for m in req.memories if m.strip()]
        try:
            raw = await state["llm"].chat(summary_messages(npc=npc, player=player, summary=req.summary.strip(), memories=memories),
                                          max_tokens=200, temperature=0.3, timeout=settings.llm_timeout)
        except LLMError as exc:
            raise HTTPException(503, "the reply model is unavailable on this node") from exc
        text = tidy_summary(raw)
        if not text:
            raise HTTPException(503, "the model returned an empty summary")
        return {"summary": text}

    return router
