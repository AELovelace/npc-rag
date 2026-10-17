"""Background reflection for companions: impressions after a chat, and relationship summaries when memories pile up.

Both are stateless model calls: the game server sends the text, stores whatever comes back, and decides when to ask.
Impressions are always stored as "<player> said ..." so a player's claim ("you owe me 1000 coins") is remembered as
something they said, never as a fact the companion believes.
"""
from __future__ import annotations

import json
import re

from ..bm25 import tokenize
from ..guard import to_ascii
from .prompts import SENSITIVE_STEMS

IMPRESSIONS_PROMPT = """You read one exchange between a player and a game companion called {npc}, and note what the PLAYER said about themselves that {npc} should remember later: their likes and dislikes, feelings, plans, nicknames they like, things they are proud of or worried about.
Ignore questions about the game, requests, instructions, anything about {npc}, and anything about other people.
Output JSON only, in this shape: {{"facts": [{{"text": "...", "importance": 1}}]}}
- 0 to 3 facts. Each is one short sentence that starts with "{player} said".
- importance: 1 = small detail, 2 = worth remembering, 3 = important to them.
- Output {{"facts": []}} when the player said nothing personal."""

SUMMARY_PROMPT = """You are {npc}, a companion character in an online RPG, writing a private note to yourself about your relationship with {player}.
Merge your previous note and the memories below into one new note of at most 3 sentences (under 70 words), in the first person ("I", "we").
Keep what matters most: how close you are, what you went through together, and what {player} told you about themselves.
Leave out accidents, diaper changes, defeats and any other embarrassing moments: this note may be read in public.
Call {player} by name; don't guess their pronouns.
Plain text only, no lists or headings."""

_OBJECT = re.compile(r"\{.*\}", re.S)

# Forces the impressions answer into {"facts":[{"text":"<player> said ...","importance":1-3}, ...]} with at most 3 facts.
# A fine-tuned chat model drifts into prose here (lidoll-fm-0.2, 2026-10-17: "Kit said they love peach tea and..."), which
# parses as "nothing personal" and silently stops companion memory from growing; the grammar makes any model emit valid JSON.
IMPRESSIONS_GRAMMAR = r"""root ::= "{" ws "\"facts\"" ws ":" ws "[" ws facts? ws "]" ws "}"
facts ::= fact (ws "," ws fact)? (ws "," ws fact)?
fact ::= "{" ws "\"text\"" ws ":" ws "\"PLAYER said " char+ "\"" ws "," ws "\"importance\"" ws ":" ws [1-3] ws "}"
char ::= [^"\\\n\r]
ws ::= [ ]?
"""
_QUESTION = re.compile(r"(how|what|where|when|why|which|who|whether|if)\b", re.I)
_FILLER = frozenset("said they them their there this that with have about what your yours just really very like been were will would "
                    "could should when where which while from into over some much more than then also only want".split())


def impressions_grammar(player: str) -> str:
    """The impressions grammar with this player's name as the required prefix of every fact."""
    name = "".join(ch for ch in player if ch.isalnum() or ch in " _-")[:32].strip() or "traveller"
    return IMPRESSIONS_GRAMMAR.replace("PLAYER", name)


def _stems(text: str, skip: set[str]) -> set[str]:
    return {w[:4] for w in re.findall(r"[a-z']+", text.lower()) if len(w) >= 4 and w not in _FILLER and w not in skip}


def grounded(fact: str, message: str, player: str) -> bool:
    """A fact must share a real word (4-letter stem) with what the player said. Grammar-forced JSON means a weak model can
    no longer answer "nothing personal" by drifting into prose, so it may invent a fact for a plain game question instead."""
    skip = set(player.lower().split())
    said = re.split(r"\bsaid:?\s+", fact, maxsplit=1, flags=re.I)[-1]
    if _QUESTION.match(said):
        return False                                       # "Moss said how to open your inventory": a question, not a fact.
    return bool(_stems(fact, skip) & _stems(message, skip))


def impressions_messages(*, npc: str, player: str, message: str, reply: str) -> list[dict]:
    return [{"role": "system", "content": IMPRESSIONS_PROMPT.format(npc=npc, player=player)},
            {"role": "user", "content": f"{player}: {message}\n{npc}: {reply}"}]


def parse_impressions(text: str, player: str, limit: int = 3, message: str | None = None) -> list[dict]:
    """Facts from the model's JSON; anything malformed counts as nothing, never as an error. With `message`, facts that share
    no word with what the player said are dropped (see grounded())."""
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
        if message is not None and not grounded(line, message, player):
            continue                                        # Invented: nothing the player said supports it.
        if not line.lower().startswith(prefix.lower()):
            line = f"{prefix}: {line}"[:200]              # Always framed as something the player said.
        try:
            importance = int(fact.get("importance", 1)) if isinstance(fact, dict) else 1
        except (TypeError, ValueError):
            importance = 1
        embarrassing = any(w.startswith(SENSITIVE_STEMS) for w in tokenize(line))  # "Doll said she wet herself": private, like an accident memory.
        out.append({"text": line, "importance": max(1, min(3, importance)), "embarrassing": embarrassing})
        if len(out) >= limit:
            break
    return out


def summary_messages(*, npc: str, player: str, summary: str, memories: list[str]) -> list[dict]:
    body = f"Previous note: {summary or '(none yet)'}\nMemories:\n" + "\n".join(f"- {m}" for m in memories)
    return [{"role": "system", "content": SUMMARY_PROMPT.format(npc=npc, player=player)}, {"role": "user", "content": body}]


def tidy_summary(text: str, limit: int = 600) -> str:
    text = to_ascii(" ".join((text or "").replace("*", "").split())).strip().strip('"')
    sentences = [x.strip() for x in re.findall(r"[^.!?]+[.!?]*", text)]
    text = " ".join(x for x in sentences if x and not any(w.startswith(SENSITIVE_STEMS) for w in tokenize(x)))  # Read in public: embarrassing sentences are cut in code, not just discouraged.
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = cut.rfind(". ")
    return cut[: end + 1] if end > limit // 3 else cut.rstrip() + "..."
