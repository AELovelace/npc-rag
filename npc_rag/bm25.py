"""Small BM25 keyword ranker: catches exact game words (Cursebreaker, RPP, Honeydew) that embeddings can blur."""
from __future__ import annotations

import math
import re
from collections import Counter

# Common English words that carry no meaning for retrieval.
STOPWORDS = frozenset(
    "a about an and are as at be been but by can could do does for from get got has have how i if in into is it its "
    "me my of on or so than that the their them then there these they this to up was we what when where which who "
    "why will with would you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords; a trailing plural 's' is trimmed so 'diapers' matches 'diaper'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words if w not in STOPWORDS]


class BM25:
    """Okapi BM25 over pre-tokenized documents."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]           # Term counts per document.
        self.lengths = [sum(c.values()) for c in self.docs]             # Document lengths in tokens.
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        df: Counter = Counter()
        for counts in self.docs:
            df.update(counts.keys())                                    # Document frequency of every term.
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}  # Always positive IDF.

    def scores(self, query: str) -> list[float]:
        """BM25 score of every document for the query (0 when no term matches)."""
        terms = tokenize(query)
        out = []
        for counts, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                tf = counts.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                    s += self.idf[t] * tf * (self.k1 + 1) / norm
            out.append(s)
        return out
