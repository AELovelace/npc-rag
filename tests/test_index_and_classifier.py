import asyncio

import pytest

from npc_rag.bm25 import BM25, tokenize
from npc_rag.classifier import GAME, GENERAL, Classifier, parse_tiebreak
from npc_rag.embedder import HashEmbedder
from npc_rag.index import WikiIndex


def test_tokenize_drops_stopwords_and_plurals():
    assert tokenize("Where are the Toilets?") == ["toilet"]


def test_bm25_prefers_matching_document():
    bm = BM25(["toilets and outhouses", "cursed items and the cursebreaker"])
    s = bm.scores("cursebreaker")
    assert s[1] > 0 and s[0] == 0


def test_search_finds_the_right_section(index):
    hits, best = index.search("cursebreaker removes cursed item", 3)
    assert hits[0].chunk.id == "needs#cursed-equipment:0"
    assert best == pytest.approx(max(h.dense for h in hits), abs=1e-6)


def test_index_round_trip_and_model_check(index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = WikiIndex.load(tmp_path / "idx", index.embedder)
    assert [c.id for c in loaded.chunks] == [c.id for c in index.chunks]

    class Other(HashEmbedder):
        name = "other-model"
    with pytest.raises(ValueError, match="rebuild"):
        WikiIndex.load(tmp_path / "idx", Other())


def _clf(embedder, **kw):
    return Classifier(embedder, ["where is the toilet", "how do I remove a cursed item"],
                      ["hello there friend", "tell me a joke please"], margin=0.2, wiki_evidence=0.5, **kw)


def test_knn_decides_clear_cases(embedder):
    v = asyncio.run(_clf(embedder).classify("where is the nearest toilet"))
    assert (v.category, v.method) == (GAME, "knn")
    v = asyncio.run(_clf(embedder).classify("hello friend"))
    assert (v.category, v.method) == (GENERAL, "knn")


def test_unsure_messages_use_wiki_then_llm(embedder):
    async def strong(_):
        return 0.9

    async def weak(_):
        return 0.1

    async def llm_says_chat(_):
        return GENERAL

    v = asyncio.run(_clf(embedder, evidence=strong).classify("purple banana"))
    assert (v.category, v.method) == (GAME, "wiki")
    v = asyncio.run(_clf(embedder, evidence=weak, tiebreak=llm_says_chat).classify("purple banana"))
    assert (v.category, v.method) == (GENERAL, "llm")
    v = asyncio.run(_clf(embedder, evidence=weak).classify("purple banana"))
    assert v.method == "knn-weak"


def test_parse_tiebreak():
    assert parse_tiebreak("GAME") == GAME
    assert parse_tiebreak(" chat.") == GENERAL
    assert parse_tiebreak("maybe") is None and parse_tiebreak("") is None
