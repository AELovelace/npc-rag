import asyncio

import pytest

from npc_rag.bm25 import BM25, tokenize
from npc_rag.classifier import GAME, GENERAL, classify, parse_verdict
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


def _run(answer, wiki_best):
    async def ask(_):
        return answer

    async def evidence(_):
        return wiki_best

    return asyncio.run(classify("purple banana", ask_llm=ask, evidence=evidence, wiki_evidence=0.5))


def test_classifier_model_decides_when_the_wiki_is_unsure():
    assert (_run(GENERAL, 0.3).category, _run(GENERAL, 0.3).method) == (GENERAL, "llm")
    assert (_run(GAME, 0.3).category, _run(GAME, 0.3).method) == (GAME, "llm")


def test_strong_wiki_match_overrides_chat():
    v = _run(GENERAL, 0.9)
    assert (v.category, v.method) == (GAME, "wiki")
    assert _run(GAME, 0.9).method == "llm"


def test_wiki_score_decides_when_the_model_is_down():
    assert (_run(None, 0.9).category, _run(None, 0.9).method) == (GAME, "wiki")
    assert _run(None, 0.1).category == GENERAL
    assert _run(None, None).category == GENERAL  # No model, no index: treat it as small talk.


def test_parse_verdict():
    assert parse_verdict("GAME") == GAME
    assert parse_verdict(" chat.") == GENERAL
    assert parse_verdict("**Game**") == GAME
    assert parse_verdict("maybe") is None and parse_verdict("") is None
