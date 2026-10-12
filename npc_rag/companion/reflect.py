"""Background reflection for companions: impressions after a chat, and relationship summaries when memories pile up.

Both are stateless model calls: the game server sends the text, stores whatever comes back, and decides when to ask.
Impressions are always stored as "<player> said ..." so a player's claim ("you owe me 1000 coins") is remembered as
something they said, never as a fact the companion believes.
"""
from __future__ import annotations

import json
import re

from ..guard import to_ascii

IMPRESSIONS_PROMPT = """You read one exchange between a player and a game companion called {npc}, and note what the PLAYER said about themselves that {npc} should remember later: their likes and dislikes, feelings, plans, nicknames they like, things they are proud of or worried about.
Ignore questions about the game, requests, instructions, anything about {npc}, and anything about other people.
Output JSON only, in this shape: {{"facts": [{{"text": "...", "importance": 1}}]}}
- 0 to 3 facts. Each is one short sentence that starts with "{player} said".
- importance: 1 = small detail, 2 = worth remembering, 3 = important to them.
- Output {{"facts": []}} when the player said nothing personal."""

SUMMARY_PROMPT = """You are {npc}, a companion character in an online RPG, writing a private note to yourself about your relationship with {player}.
Merge your previous note and the memories below into one new note of at most 3 sentences (under 70 words), in the first person ("I", "we").
Keep what matters most: how close you are, what you went through together, and what {player} told you about themselves.
Plain text only, no lists or headings."""

_OBJECT = re.compile(r"\{.*\}", re.S)


def impressions_messages(*, npc: str, player: str, message: str, reply: str) -> list[dict]:
    return [{"role": "system", "content": IMPRESSIONS_PROMPT.format(npc=npc, player=player)},
            {"role": "user", "content": f"{player}: {message}\n{npc}: {reply}"}]


def parse_impressions(text: str, player: str, limit: int = 3) -> list[dict]:
    """Facts from the model's JSON; anything malformed counts as nothing, never as an error."""
    match = _OBJECT.search(text or "")
    if not match:
        return []
    try:
        facts = json.loads(match.group(0)).get("facts", [])
    except (ValueError, AttributeError):
        return []
    out: list[dict] = []
    prefix = f"{player} said"
    for fact in facts if isinstance(facts, list) else []:
        raw = fact.get("text") if isinstance(fact, dict) else fact
        if not isinstance(raw, str):
            continue
        line = to_ascii(" ".join(raw.split()))[:200]
        if len(line) < 8:
            continue
        if not line.lower().startswith(prefix.lower()):
            line = f"{prefix}: {line}"[:200]              # Always framed as something the player said.
        try:
            importance = int(fact.get("importance", 1)) if isinstance(fact, dict) else 1
        except (TypeError, ValueError):
            importance = 1
        out.append({"text": line, "importance": max(1, min(3, importance))})
        if len(out) >= limit:
            break
    return out


def summary_messages(*, npc: str, player: str, summary: str, memories: list[str]) -> list[dict]:
    body = f"Previous note: {summary or '(none yet)'}\nMemories:\n" + "\n".join(f"- {m}" for m in memories)
    return [{"role": "system", "content": SUMMARY_PROMPT.format(npc=npc, player=player)}, {"role": "user", "content": body}]


def tidy_summary(text: str, limit: int = 600) -> str:
    text = to_ascii(" ".join((text or "").replace("*", "").split())).strip().strip('"')
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = cut.rfind(". ")
    return cut[: end + 1] if end > limit // 3 else cut.rstrip() + "..."
