"""Companion backstories (lore) pushed by the game server, plus a small embedding cache for memory texts.

Lore is tiny (a handful of sections per companion), so each companion keeps a plain matrix of section vectors
and every search is one dot product. Secrets carry a reveal gate: a minimum bond and/or a story flag the
player must have before the section can be retrieved at all.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LoreSection:
    npc: str                    # Companion id (followers-data.json key), e.g. "merchant_mira".
    id: str                     # Stable section id, returned in responses so the game server can log what was used.
    heading: str
    text: str
    reveal_bond: int = 0        # Minimum bond (0-100) before this section can be used.
    reveal_flag: str = ""       # Story flag the player must have set ("" = none).
    draft: bool = False         # Lumi's drafts are live until a GM signs them off; the game server decides whether to send them.

    def embed_text(self) -> str:
        return f"{self.heading}\n{self.text}"

    def unlocked(self, bond: int, flags: set[str]) -> bool:
        return bond >= self.reveal_bond and (not self.reveal_flag or self.reveal_flag in flags)  # Locked secrets are never even scored.


@dataclass
class LoreStore:
    version: str                                            # The game server's hash of the payload; chat requests must match it.
    sections: dict[str, list[LoreSection]] = field(default_factory=dict)
    vectors: dict[str, np.ndarray] = field(default_factory=dict)  # One unit row per section, same order as sections[npc].

    @classmethod
    def build(cls, version: str, sections: list[LoreSection], embedder) -> "LoreStore":
        store = cls(version)
        for s in sections:
            store.sections.setdefault(s.npc, []).append(s)
        for npc, items in store.sections.items():
            store.vectors[npc] = embedder.passages([s.embed_text() for s in items])  # One batch per companion.
        return store

    def count(self) -> int:
        return sum(len(v) for v in self.sections.values())

    def search(self, npc: str, qv: np.ndarray, bond: int, flags: set[str], k: int, min_score: float) -> list[tuple[LoreSection, float]]:
        """Best unlocked sections for this companion, most relevant first."""
        items = self.sections.get(npc) or []
        if not items or k <= 0:
            return []
        scores = self.vectors[npc] @ qv                      # Cosine similarity with every section of this companion.
        ranked = sorted(((items[i], float(scores[i])) for i in range(len(items)) if items[i].unlocked(bond, flags)),
                        key=lambda p: p[1], reverse=True)
        return [(s, score) for s, score in ranked if score >= min_score][:k]


class EmbedCache:
    """LRU cache of passage vectors keyed by text hash: the same memories arrive with every chat."""

    def __init__(self, embedder, max_items: int = 5000):
        self.embedder = embedder
        self.max_items = max_items
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def passages(self, texts: list[str]) -> np.ndarray:
        """Vectors for texts (unit rows), embedding only the ones not seen recently."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        keys = [self._key(t) for t in texts]
        missing = list(dict.fromkeys(i for i, k in enumerate(keys) if k not in self._data))  # Unique misses, in order.
        if missing:
            fresh = self.embedder.passages([texts[i] for i in missing])
            for row, i in enumerate(missing):
                self._data[keys[i]] = fresh[row]
        out = []
        for k in keys:
            self._data.move_to_end(k)                        # Recently used stays.
            out.append(self._data[k])
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)                   # Oldest out.
        return np.vstack(out)
