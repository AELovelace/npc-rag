import asyncio

import pytest

from npc_rag.bm25 import BM25, tokenize
from npc_rag.classifier import GAME, GENERAL, Verdict, classify, parse_verdict
from npc_rag.embedder import HashEmbedder
from npc_rag.index import WikiIndex


def test_tokenize_drops_stopwords_and_plurals():
    assert tokenize("Where are the Toilets?") == ["toilet"]


def test_bm25_prefers_matching_document():
    bm = BM25(["toilets and outhouses", "cursed items and the cursebreaker"])
    s = bm.scores("cursebreaker")
    assert s[1] > 0 and s[0] == 0


def test_bm25_coverage_weighs_rare_and_unknown_words():
    bm = BM25(["toilets and outhouses", "cursed items and the cursebreaker", "cursed boots"])
    assert bm.coverage("where is the cursebreaker", 1) == pytest.approx(1.0)  # Every meaningful word matched.
    assert 0 < bm.coverage("cursebreaker everest", 1) < 0.5  # An unknown word outweighs even the rarest wiki word.
    assert 0 < bm.coverage("cursed cursebreaker", 2) < 0.5  # The common word matched, the rare one didn't.
    assert bm.coverage("how are you", 1) == 0.0  # Stopwords only: no keyword evidence.


def test_search_evidence_reports_keyword_coverage(index):
    hits, best, coverage = index.search_evidence("cursebreaker", 3)
    plain_hits, plain_best = index.search("cursebreaker", 3)
    assert [h.chunk.id for h in hits] == [h.chunk.id for h in plain_hits] and best == plain_best
    assert coverage == pytest.approx(1.0)
    assert index.search_evidence("zzzz qqqq", 3)[2] == 0.0  # No shared words at all.


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


def _run_keyword(answer, dense, coverage, *, rule=(0.5, 0.4)):
    async def ask(_):
        return answer

    async def evidence(_):
        return {"dense": dense, "coverage": coverage}

    return asyncio.run(classify("purple banana", ask_llm=ask, evidence=evidence, wiki_evidence=0.5,
                                keyword_coverage=rule[0] if rule else None, keyword_dense=rule[1] if rule else None))


def test_keyword_coverage_lowers_the_override_bar():
    v = _run_keyword(GENERAL, 0.45, 0.8)  # Below RAG_GAME_EVIDENCE, but the wiki holds most of the wording.
    assert (v.category, v.method) == (GAME, "wiki")
    assert v.scores == {"llm": GENERAL, "wiki_best": 0.45, "wiki_keyword": 0.8}
    assert _run_keyword(GENERAL, 0.45, 0.3).category == GENERAL  # Too little of the message is in the wiki.
    assert _run_keyword(GENERAL, 0.35, 0.9).category == GENERAL  # Coverage never overrides a weak dense score.
    assert _run_keyword(GENERAL, 0.45, 0.9, rule=None).category == GENERAL  # Unset thresholds disable the rule.
    assert _run_keyword(None, 0.45, 0.8).category == GAME  # Also the fallback while the model is down.


def test_wiki_first_skips_the_model_when_the_wiki_decides():
    asked = []

    async def ask(message):
        asked.append(message)
        return GENERAL

    def run(dense):
        async def evidence(_):
            assert not asked  # The model must not start before the wiki answers.
            return dense
        return asyncio.run(classify("purple banana", ask_llm=ask, evidence=evidence, wiki_evidence=0.5, wiki_first=True))

    strong = run(0.9)
    assert (strong.category, strong.method, strong.scores["llm"], asked) == (GAME, "wiki", None, [])
    weak = run(0.3)
    assert (weak.category, weak.method, asked) == (GENERAL, "llm", ["purple banana"])  # Weak evidence still asks once.


def test_verdict_brief_for_log_lines():
    assert Verdict(GAME, "wiki", {"llm": GENERAL, "wiki_best": 0.71234, "wiki_keyword": 0.5}).brief() ==         "llm=general wiki_best=0.7123 wiki_keyword=0.5"
    assert Verdict(GAME, "caller").brief() == "-"


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
