"""Code-level guards that keep Pip's replies tied to the wiki.

Small models drift even with careful prompts: they invent names, answer from a weakly
matching section, or recite their instructions. These checks catch that in code:

  select_hits            - drop sections below MIN_RELEVANCE or far below the best match
  find_unsupported_terms - capitalised mid-sentence words that no source text mentions
  leaks_persona          - long word runs copied from the rules/persona
  to_ascii               - curly quotes, dashes and accents flattened for GameMaker fonts
"""
from __future__ import annotations

import re
import unicodedata

# Capitalised words that are fine mid-sentence even when no source mentions them.
BASE_ALLOWED = {
    "i", "i'm", "i'll", "i've", "i'd", "ok", "okay", "ai", "npc", "oh", "hi", "hey", "hello",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

ASCII_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "–": "-", "—": "-", "―": "-", "−": "-", "‐": "-", "‑": "-",
    "…": "...", " ": " ", " ": " ", " ": " ", "·": "-", "•": "-",
}

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")
# Text before a word that puts it at the start of a sentence or list item (optionally behind a quote or bold).
_SENTENCE_START = re.compile(r"(^\s*|[.!?:][\"')\]*”’]*\s+|\n[\s*\-\d.)]*)[\"'(\[*“‘]*$")


def select_hits(hits: list[dict], min_relevance: float, keep_margin: float, limit: int, min_words: int = 0) -> list[dict]:
    """Sections worth showing the model: above MIN_RELEVANCE, within keep_margin of the best one, and
    long enough to answer something (a one-line chapter overview invites the model to fill in the rest)."""
    kept = sorted((h for h in hits if h["dense"] >= min_relevance and len(h["text"].split()) >= min_words),
                  key=lambda h: h["dense"], reverse=True)
    if kept:
        floor = kept[0]["dense"] - keep_margin  # A weak runner-up (Utopia for a diaper question) can't steer the answer.
        kept = [h for h in kept if h["dense"] >= floor]
    return kept[:limit]


def to_ascii(text: str) -> str:
    mapped = "".join(ASCII_MAP.get(ch, ch) for ch in text)
    return unicodedata.normalize("NFKD", mapped).encode("ascii", "ignore").decode("ascii")


def _norm(word: str) -> str:
    word = word.lower().replace("’", "'").strip("'-")
    return word[:-2] if word.endswith("'s") else word


def _vocabulary(texts: list[str]) -> set[str]:
    vocab = set(BASE_ALLOWED)
    for text in texts:
        for m in _WORD.finditer(text):
            word = _norm(m.group(0))
            if word:
                vocab.add(word)
                vocab.update(p for p in re.split(r"[-']", word) if p)  # "Cursebreaker's" / "pay-toilet" parts.
    return vocab


def _known(word: str, vocab: set[str]) -> bool:
    # Simple plural/singular variants count ("dragons" vs "dragon", "gods" vs "god").
    return (word in vocab or word + "s" in vocab or (word.endswith("s") and word[:-1] in vocab)
            or (word.endswith("es") and word[:-2] in vocab))


def find_unsupported_terms(reply: str, sources: list[str]) -> list[str]:
    """Capitalised words in the middle of a sentence that appear in none of the sources.

    Sentence-initial words are skipped (every sentence starts with a capital), so an
    invented name that only ever opens a sentence slips through.
    """
    vocab = _vocabulary(sources)
    found: dict[str, str] = {}
    for m in _WORD.finditer(reply):
        token = m.group(0)
        if not token[0].isupper() or _SENTENCE_START.search(reply[: m.start()]):
            continue
        word = _norm(token)
        if not word or word.isdigit():
            continue
        parts = [p for p in re.split(r"[-']", word) if p]
        if _known(word, vocab) or (parts and all(_known(p, vocab) for p in parts)):
            continue
        found.setdefault(word, token.rstrip("'’-"))
    return list(found.values())


def _shingles(text: str, size: int) -> set[tuple[str, ...]]:
    words = [w for w in (_norm(m.group(0)) for m in _WORD.finditer(text)) if w]
    return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}


def leaks_persona(reply: str, persona: str, ngram: int = 6, min_hits: int = 3) -> bool:
    """True when the reply shares at least `min_hits` runs of `ngram` words with the rules/persona."""
    return bool(persona.strip()) and len(_shingles(reply, ngram) & _shingles(persona, ngram)) >= min_hits


def retry_rule(terms: list[str]) -> str:
    """Extra system-prompt line for the second attempt after a reply named unsupported things."""
    return ("\nYour previous answer mentioned " + ", ".join(terms) + ", which the guidebook never mentions. "
            "Do not mention them. Only use names that appear above; if the guidebook doesn't answer the "
            "question, say you're not sure.")
