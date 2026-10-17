"""The companion reply pipeline as a LangGraph StateGraph.

    START ─┬─ classify ── wiki ───────┐
           └─ embed ──┬─ lore ────────┼─ compose ─┬─ generate ─ finish ─ END
                      └─ memories ────┘           └─────────────┘ (game question with nothing to go on: no model call)

classify  - caller verdict, else Sakura's classifier (9091) with the wiki score as backup (same as Pip)
wiki      - game questions get the best wiki sections (same index and guard rules as Pip)
embed     - one query vector for the message, shared by lore and memories
lore      - this companion's unlocked backstory sections (bond + story-flag gates); a {"secret":"locked"} marker
            when the message is about a section still behind its gate (never its text)
memories  - rank the game server's candidate memories by meaning, importance and age; anonymous gossip
            only when no own memory fits; embarrassing memories stay out while others can hear (slips only)
compose   - build the prompt; a slip candidate is added only when someone else is listening
generate  - main model (9090) + the existing guard (invented names retried, recited rules blocked)
finish    - tidy for the dialogue box, and confirm whether the slip was really said
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..classifier import GAME, GENERAL, Verdict
from ..guard import select_hits, to_ascii
from ..prompts import tidy_reply, unsure_reply
from .prompts import AMBIENT_KINDS, PAID_KINDS, build_messages, copied, mentions, rules_text, strip_speaker_labels

FOLLOW_UP_WORDS = 5  # Same as Pip: a short follow-up is searched together with the previous question.


def _merge(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}  # Parallel branches each add their own timing.


class CompanionState(TypedDict, total=False):
    req: Any                                  # The validated CompanionChatRequest.
    message: str
    history: list[dict]                       # OpenAI roles, newest last.
    npc_name: str
    player_name: str
    category: str
    method: str
    scores: str                               # Classifier signals for the log line (Verdict.brief()).
    cache: dict                               # The classifier's wiki search, reused by the wiki node.
    notes: list[dict]
    hits: list[dict]
    qv: Any                                   # Query vector (numpy row).
    lore: list                                # [(LoreSection, score)]
    locked: bool                              # The message is about a section still behind its gate (marker only).
    memories: list[dict]                      # Chosen own memories: {id, text}.
    gossip: dict | None                       # Chosen anonymous gossip line: {id, text}.
    slip: dict | None                         # Slip candidate offered to the model: {id, text}.
    messages: list[dict]
    rules: str
    raw: str | None
    guard: str
    reply: str
    slipped: bool
    timings: Annotated[dict, _merge]


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def build_companion_graph(settings, state: dict, *, search, classify_with_search, generate, embed_cache):
    """Compile the graph once per app; the callables come from service.create_app so Pip and companions share them."""

    async def classify(s: CompanionState) -> dict:
        t0 = time.perf_counter()
        req, cache = s["req"], {}
        if req.kind != "chat":
            verdict = Verdict(GENERAL, "kind")                     # Emotes, reactions, musings and banter are never game questions.
        elif req.category:
            verdict = Verdict(GAME if req.category == "game" else GENERAL, "caller")  # The game server already knows.
        else:
            verdict = await classify_with_search(s["message"], cache)
        return {"category": verdict.category, "method": verdict.method, "scores": verdict.brief(),
                "cache": cache, "timings": {"classify": _ms(t0)}}

    async def wiki(s: CompanionState) -> dict:
        if s["category"] != GAME:
            return {"notes": [], "hits": []}
        t0 = time.perf_counter()
        message, history = s["message"], s["history"]
        last_user = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
        query = f"{last_user} {message}" if last_user and len(message.split()) < FOLLOW_UP_WORDS else message
        result = s["cache"].get("search") if query == message else None
        hits = (result or await search(query, settings.top_k))["results"]
        notes = select_hits(hits, settings.min_relevance, settings.guard_keep_margin, settings.companion_note_k, settings.guard_min_words)
        notes = [{**n, "text": " ".join(n["text"].split()[: settings.companion_note_words])} for n in notes]  # Shorter prompts read faster.
        return {"notes": notes, "hits": hits, "timings": {"retrieve": _ms(t0)}}

    async def embed(s: CompanionState) -> dict:
        t0 = time.perf_counter()
        qv = (await asyncio.to_thread(state["embedder"].queries, [s["message"]]))[0]  # CPU work off the event loop.
        return {"qv": qv, "timings": {"embed": _ms(t0)}}

    async def lore(s: CompanionState) -> dict:
        store, req = state.get("lore"), s["req"]
        if store is None:
            return {"lore": []}
        found = store.search(req.npc_id, s["qv"], req.bond, set(req.flags), settings.companion_lore_k, settings.companion_lore_min)
        locked = req.kind in ("chat", "emote") and store.locked_match(
            req.npc_id, s["qv"], req.bond, set(req.flags), settings.companion_locked_min) is not None  # Asked about a gated secret: deflect, don't deny.
        return {"lore": found, "locked": locked}

    async def memories(s: CompanionState) -> dict:
        req, qv = s["req"], s["qv"]
        t0 = time.perf_counter()
        listening = req.audience > 0
        pool = [m for m in req.memories if not (m.embarrassing and listening)]  # Embarrassing ones only come out as slips.
        chosen: list[dict] = []
        if pool:
            vecs = await asyncio.to_thread(embed_cache.passages, [m.text for m in pool])
            rel = vecs @ qv
            scored = []
            for m, r in zip(pool, rel):
                r = float(r)
                if r < settings.companion_memory_min:
                    continue                                      # Unrelated memories never crowd the prompt.
                score = 0.6 * r + 0.25 * (m.importance / 5) + 0.15 * math.exp(-m.age_days / 30)  # Meaning, then importance, then freshness.
                scored.append((score, m))
            scored.sort(key=lambda p: p[0], reverse=True)
            chosen = [{"id": m.id, "text": m.text} for _, m in scored[: settings.companion_memory_k]]
        gossip = None
        if not chosen and req.gossip:                             # Gossip only fills a silence; it never beats a shared memory.
            gvecs = await asyncio.to_thread(embed_cache.passages, [g.text for g in req.gossip])
            grel = gvecs @ qv
            best = int(grel.argmax())
            if float(grel[best]) >= settings.companion_gossip_min:
                gossip = {"id": req.gossip[best].id, "text": req.gossip[best].text}
        return {"memories": chosen, "gossip": gossip, "timings": {"memories": _ms(t0)}}

    async def compose(s: CompanionState) -> dict:
        req = s["req"]
        slip = {"id": req.slip.id, "text": req.slip.text} if req.slip and req.audience > 0 and req.kind != "banter" else None  # A slip needs listeners.
        fallback = "lives and works here." if req.kind in AMBIENT_KINDS else "is a loyal adventuring companion."
        persona = req.persona.strip() or f"{s['npc_name']} {fallback}"  # Residents are never described as companions.
        messages = build_messages(
            name=s["npc_name"], player=s["player_name"], persona=persona, voice=list(req.voice), audience=req.audience,
            category=s["category"], message=s["message"], history=s["history"], notes=s["notes"],
            lore=[sec for sec, _ in s["lore"]], summary=req.summary.strip(), memories=[m["text"] for m in s["memories"]],
            gossip=[s["gossip"]["text"]] if s["gossip"] else [], slip=slip["text"] if slip else None,
            kind=req.kind, situation=list(req.situation), about=req.about.strip(),
            partner=req.partner.model_dump() if req.partner else None, thread=[t.model_dump() for t in req.thread],
            topic=req.topic.strip(), said=list(req.said), speaking_style=req.speaking_style, locked=bool(s.get("locked")))
        rules = rules_text(name=s["npc_name"], player=s["player_name"], persona=persona, audience=req.audience, kind=req.kind, speaking_style=req.speaking_style)
        return {"messages": messages, "rules": rules, "slip": slip}

    def route(s: CompanionState) -> str:
        nothing = s["category"] == GAME and not s["notes"] and not s["lore"] and not s["memories"]
        return "finish" if nothing else "generate"              # A game question with nothing to go on never reaches the model.

    async def run_generate(s: CompanionState) -> dict:
        t0 = time.perf_counter()
        sources = [m["content"] for m in s["messages"]] + [s["npc_name"], s["player_name"], *settings.guard_allowed_terms]
        req = s["req"]
        lane = "paid" if req.kind in PAID_KINDS else "background"  # Cost guard: only PAID_KINDS may ever reach OpenAI.
        raw, verdict = await generate(s["messages"], s["rules"], sources, settings.companion_max_tokens, lane=lane)  # LLMError propagates: 503, the pool fails over.
        resident = req.kind in AMBIENT_KINDS
        thread = [t.text for t in req.thread] + list(req.said)  # Earlier conversations count as "already said" too.
        check = lambda text: copied(text, list(req.voice), req.partner.line if req.partner else "", thread, 6 if resident else 7, 7 if resident else 5)  # Residents answering each other naturally reuse a few of their words (live 2026-10-15: half of Felix's replies were dropped at 5).  # Residents: 5 words of a voice line is a recital.
        what = check(raw or "") if raw and verdict.startswith("passed") else None
        if what:  # One retry with the copy named; keep the first reply if the retry fails the guard.
            note = {**s["messages"][0], "content": s["messages"][0]["content"] + f"\nYour last reply repeated {what}. Say something new in your own words."}
            raw2, verdict2 = await generate([note, *s["messages"][1:]], s["rules"], sources, settings.companion_max_tokens, lane=lane)
            if raw2 and verdict2.startswith("passed") and not (resident and check(raw2)):
                raw, verdict = raw2, "passed_after_retry"
            elif resident:
                raw, verdict = None, "repeated"  # A resident that still repeats itself says nothing (the game server drops it).
        return {"raw": raw, "guard": verdict, "timings": {"generate": _ms(t0)}}

    async def finish(s: CompanionState) -> dict:
        raw = s.get("raw")
        if not s.get("guard"):                                   # route() skipped the model: nothing to go on.
            reply, verdict = unsure_reply(s["category"], s["hits"]), "no_relevant_notes"
        elif raw is None:
            reply, verdict = unsure_reply(s["category"], s["hits"]), s["guard"]
        else:
            req = s["req"]
            if req.kind in AMBIENT_KINDS:                      # Residents: drop "Name:" labels and stage directions small models add.
                raw = strip_speaker_labels(raw, [s["npc_name"], s["player_name"], req.partner.name if req.partner else "", *(t.speaker for t in req.thread)])
            reply, verdict = tidy_reply(to_ascii(raw), s["npc_name"], settings.companion_max_reply_chars), s["guard"]  # ASCII first: "..." for an ellipsis can't push a tidied reply past the limit.
        reply = to_ascii(reply)                                   # GameMaker fonts often lack curly quotes and dashes.
        names = set(s["npc_name"].lower().split()) | set(s["player_name"].lower().split())
        slipped = bool(s.get("slip")) and verdict.startswith("passed") and mentions(reply, s["slip"]["text"], names, s["player_name"])
        return {"reply": reply, "guard": verdict, "slipped": slipped}

    g = StateGraph(CompanionState)
    for name, fn in (("classify", classify), ("wiki", wiki), ("embed", embed), ("lore", lore), ("memories", memories),
                     ("compose", compose), ("generate", run_generate), ("finish", finish)):
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_edge(START, "embed")
    g.add_edge("classify", "wiki")
    g.add_edge("embed", "lore")
    g.add_edge("embed", "memories")
    g.add_edge(["wiki", "lore", "memories"], "compose")      # Waits for all three branches.
    g.add_conditional_edges("compose", route, {"generate": "generate", "finish": "finish"})
    g.add_edge("generate", "finish")
    g.add_edge("finish", END)
    return g.compile()
