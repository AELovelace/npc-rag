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


def test_strong_wiki_returns_without_waiting_for_model_and_cleans_up():
    async def run():
        started, cancelled = asyncio.Event(), asyncio.Event()
        cache = {}

        async def ask(_):
            started.set()
            try:
                await asyncio.Event().wait()  # A saturated model never responds during this test.
            finally:
                cancelled.set()

        async def evidence(_):
            await started.wait()
            cache["notes"] = "Cursebreaker"  # Retrieval must finish before the caller receives its verdict.
            return 0.9

        verdict = await asyncio.wait_for(classify("cursed item", ask_llm=ask, evidence=evidence,
                                                 wiki_evidence=0.5), timeout=1)
        assert (verdict.category, verdict.method) == (GAME, "wiki")
        assert verdict.scores == {"llm": None, "wiki_best": 0.9}
        assert cache["notes"] == "Cursebreaker" and cancelled.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("wiki_best, expected", [(0.1, GENERAL), (0.9, GAME)])
def test_fast_model_still_waits_for_wiki_override(wiki_best, expected):
    async def run():
        model_done = asyncio.Event()

        async def ask(_):
            model_done.set()
            return GENERAL

        async def evidence(_):
            await model_done.wait()
            await asyncio.sleep(0)  # Deliver the wiki evidence after the model's CHAT answer.
            return wiki_best

        verdict = await classify("message", ask_llm=ask, evidence=evidence, wiki_evidence=0.5)
        assert verdict.category == expected

    asyncio.run(run())


def test_weak_wiki_waits_for_model():
    async def run():
        wiki_done = asyncio.Event()

        async def evidence(_):
            wiki_done.set()
            return 0.1

        async def ask(_):
            await wiki_done.wait()
            await asyncio.sleep(0)
            return GAME

        verdict = await classify("message", ask_llm=ask, evidence=evidence, wiki_evidence=0.5)
        assert (verdict.category, verdict.method) == (GAME, "llm")

    asyncio.run(run())


@pytest.mark.parametrize("cancel_caller", [False, True])
def test_classifier_cleans_up_on_search_error_or_caller_cancellation(cancel_caller):
    async def run():
        model_started, wiki_started = asyncio.Event(), asyncio.Event()
        model_stopped, wiki_stopped = asyncio.Event(), asyncio.Event()

        async def ask(_):
            model_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                model_stopped.set()

        async def evidence(_):
            wiki_started.set()
            try:
                await model_started.wait()
                if not cancel_caller:
                    raise RuntimeError("search failed")
                await asyncio.Event().wait()
            finally:
                wiki_stopped.set()

        task = asyncio.create_task(classify("message", ask_llm=ask, evidence=evidence, wiki_evidence=0.5))
        await asyncio.wait_for(wiki_started.wait(), timeout=1)
        if cancel_caller:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel_caller else RuntimeError):
            await asyncio.wait_for(task, timeout=1)
        assert model_stopped.is_set() and wiki_stopped.is_set()

    asyncio.run(run())
