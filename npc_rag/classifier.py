"""Decide whether a player message is a game question (use the wiki) or general conversation.

Three steps, cheapest first; the first confident one wins:
  1. kNN   - compare the message with labelled examples (data/classifier_seeds.json). ~10 ms on CPU.
  2. wiki  - if unsure, ask the RAG service how strongly the wiki covers the message.
  3. llm   - if still unsure, ask llama.cpp for a one-word verdict (optional).
If every step is unsure or unavailable, the kNN leaning decides ("knn-weak").
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

GAME, GENERAL = "game", "general"

TIEBREAK_PROMPT = (
    "You sort messages that players send to a tutorial character in the online RPG LiDollQuest.\n"
    "Answer GAME if the message asks how the game works: controls, rules, items, places, quests, classes, "
    "needs, accidents, protection, money, other players, bugs or what to do next.\n"
    "Answer CHAT for greetings, small talk, feelings, questions about the character herself, or anything "
    "outside the game.\nReply with exactly one word: GAME or CHAT."
)


@dataclass
class Verdict:
    category: str               # "game" or "general".
    confidence: float           # 0.5 (coin flip) .. 1.0 (certain).
    method: str                 # Which step decided: knn, wiki, llm or knn-weak.
    scores: dict = field(default_factory=dict)  # Raw signals, for tuning and logs.

    def to_dict(self) -> dict:
        return {"category": self.category, "confidence": round(self.confidence, 3), "method": self.method,
                "scores": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.scores.items()}}


def load_seeds(path: Path) -> tuple[list[str], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    game, general = data.get(GAME, []), data.get(GENERAL, [])
    if len(game) < 3 or len(general) < 3:
        raise ValueError(f"{path} needs at least 3 'game' and 3 'general' examples")
    return game, general


class Classifier:
    def __init__(self, embedder, game_examples: list[str], general_examples: list[str], *,
                 margin: float = 0.04, wiki_evidence: float = 0.66, neighbours: int = 3,
                 evidence: Callable[[str], Awaitable[float | None]] | None = None,
                 tiebreak: Callable[[str], Awaitable[str | None]] | None = None):
        self.embedder = embedder
        self.game_vecs = embedder.passages(game_examples)        # Symmetric embeddings: message vs message.
        self.general_vecs = embedder.passages(general_examples)
        self.margin = margin                                     # kNN gap needed to decide on step 1.
        self.wiki_evidence = wiki_evidence                       # Wiki score that settles it as a game question.
        self.k = neighbours                                      # How many nearest examples to average.
        self.evidence = evidence                                 # async message -> best wiki score (or None).
        self.tiebreak = tiebreak                                 # async message -> "game"/"general" (or None).

    def _top_mean(self, vecs: np.ndarray, v: np.ndarray) -> float:
        sims = vecs @ v
        k = min(self.k, len(sims))
        return float(np.sort(sims)[-k:].mean())                  # Mean of the k closest examples.

    def knn(self, message: str) -> tuple[float, float, float]:
        """(game similarity, general similarity, margin); a positive margin leans towards game."""
        v = self.embedder.passages([message])[0]
        g, n = self._top_mean(self.game_vecs, v), self._top_mean(self.general_vecs, v)
        return g, n, g - n

    async def classify(self, message: str, *, evidence: Callable[[str], Awaitable[float | None]] | None = None) -> Verdict:
        """Classify one message; `evidence` overrides the default wiki check for this call only."""
        evidence = evidence or self.evidence
        g, n, margin = self.knn(message)
        scores = {"knn_game": g, "knn_general": n, "margin": margin}
        lean = GAME if margin >= 0 else GENERAL
        knn_conf = 1 / (1 + math.exp(-abs(margin) * 40))         # margin 0.04 -> 0.83, 0.1 -> 0.98.

        if abs(margin) >= self.margin:
            return Verdict(lean, knn_conf, "knn", scores)          # Step 1 is confident.

        if evidence is not None:
            best = await evidence(message)
            if best is not None:
                scores["wiki_best"] = best
                if best >= self.wiki_evidence:
                    return Verdict(GAME, 0.5 + min(0.45, (best - self.wiki_evidence) * 3 + 0.3), "wiki", scores)

        if self.tiebreak is not None:
            answer = await self.tiebreak(message)
            if answer in (GAME, GENERAL):
                scores["llm"] = answer
                return Verdict(answer, 0.8, "llm", scores)

        return Verdict(lean, knn_conf, "knn-weak", scores)      # Nothing else available: go with the lean.


def parse_tiebreak(text: str) -> str | None:
    """Map the LLM's one-word answer onto a category; anything else is treated as no answer."""
    word = text.strip().split()[0].strip(".,!:").upper() if text.strip() else ""
    return {"GAME": GAME, "CHAT": GENERAL}.get(word)
