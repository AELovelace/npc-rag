"""Decide whether a player message is a game question (use the wiki) or general conversation.

The small classifier model Sakura runs on port 9091 and a wiki search run side by side:
  wiki  - a best wiki score of RAG_GAME_EVIDENCE or more makes it a game question, whatever the
          model said (the 3B calls needs/accident questions small talk about 1 time in 10)
  llm   - otherwise the model's GAME or CHAT answer decides
  wiki  - and if the model is down or answers something else, a low wiki score means chat
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable

GAME, GENERAL = "game", "general"

CLASSIFY_PROMPT = (
    "You sort messages that players send to a tutorial character in the online RPG LiDollQuest.\n"
    "Answer GAME if the message asks how the game works: controls, rules, items, places, quests, classes, "
    "gods, needs, accidents, protection, money, other players, bugs or what to do next.\n"
    "Bathrooms, toilets, diapers, wetting, messing and clothing are game mechanics, so those are GAME. "
    "So is asking what an unfamiliar word or term means, and any technical problem with the screen or controls.\n"
    "Answer CHAT for greetings, small talk, feelings, questions about the character herself, or anything "
    "outside the game.\nReply with exactly one word: GAME or CHAT."
)


@dataclass
class Verdict:
    category: str               # "game" or "general".
    method: str                 # Which step decided: llm or wiki.
    scores: dict = field(default_factory=dict)  # Raw signals, for tuning and logs.

    def to_dict(self) -> dict:
        return {"category": self.category, "method": self.method,
                "scores": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.scores.items()}}


def parse_verdict(text: str) -> str | None:
    """Map the model's one-word answer onto a category; anything else is treated as no answer."""
    word = text.strip().split()[0].strip(".,!:*\"'").upper() if text.strip() else ""
    return {"GAME": GAME, "CHAT": GENERAL}.get(word)


async def classify(message: str, *, ask_llm: Callable[[str], Awaitable[str | None]],
                   evidence: Callable[[str], Awaitable[float | None]], wiki_evidence: float) -> Verdict:
    """ask_llm returns "game"/"general" or None; evidence returns the best wiki score or None."""
    answer, best = await asyncio.gather(ask_llm(message), evidence(message))
    scores = {"llm": answer, "wiki_best": best}
    if best is not None and best >= wiki_evidence:
        return Verdict(GAME, "wiki" if answer != GAME else "llm", scores)  # The wiki clearly covers it.
    if answer in (GAME, GENERAL):
        return Verdict(answer, "llm", scores)
    return Verdict(GENERAL, "wiki", scores)  # Model down and the wiki barely matches: small talk.
