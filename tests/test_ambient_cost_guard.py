"""Ambient NPC life (AMBIENT_NPC_LIFE_PLAN.md): resident line kinds, and the cost guard that keeps them off OpenAI.

Doll's rule (2026-10-14): ambient AI life never runs on the paid API, and it must never fill the local slots a player's
reply needs (that would push the player onto the paid API instead).
"""
import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from npc_rag import service
from npc_rag.companion.prompts import AMBIENT_KINDS, PAID_KINDS, build_messages
from npc_rag.llm import LlamaClient, LocalBusy, OpenAIClient, OverflowClient

from test_services import FakeLlama, keyword_classifier, make_client, openai_response  # noqa: F401

WREN = {"npc_id": "utopia_wren", "npc_name": "Wren", "player_name": "Doll",
        "persona": "Wren is the tavernkeeper of Utopia: sociable, nosy and fiercely protective of Felix."}


def test_resident_recall_and_gossip_are_attributed_and_scoped_to_the_current_partner():
    from npc_rag.companion.prompts import resident_context_text
    context = {"kind": "npc", "name": "Felix", "relationship": "Your trusted colleague.",
               "memories": ["Felix said: I finished the shelf."], "gossip": ["Doll said: My hobby is painting."]}
    prompt = resident_context_text(context, kind="ambient_reply", partner={"name": "Felix"}, player="", audience=2)
    assert "trusted colleague" in prompt and "Felix said:" in prompt and "Doll said:" in prompt
    assert "not an established fact" in prompt and "current conversation take priority" in prompt
    assert resident_context_text(context, kind="ambient_reply", partner={"name": "Padma"}, player="", audience=0) == ""
    player_context = {**context, "kind": "player", "name": "Doll"}
    for kind, player, audience in [("ambient_answer", "Other", 0), ("ambient_answer", "Doll", 1),
                                    ("shop_beckon", "Doll", 0), ("chat", "Doll", 0), ("ambient_open", "Doll", 0)]:
        assert resident_context_text(player_context, kind=kind, partner={"name": "Felix"}, player=player, audience=audience) == ""
    private = resident_context_text(player_context, kind="ambient_answer", partner=None, player="Doll", audience=0)
    assert "Earlier encounters" in private and "painting" not in private and "trusted colleague" not in private


def test_resident_memory_contract_is_bounded():
    from pydantic import ValidationError
    from npc_rag.companion.router import ResidentContext
    for fields in [{"memories": ["x"] * 5}, {"gossip": ["x"] * 3}, {"memories": ["x" * 401]}, {"relationship": "x" * 401}]:
        with pytest.raises(ValidationError):
            ResidentContext(kind="npc", name="Felix", **fields)


def test_resident_memory_reaches_local_generation(settings, embedder, index, clf):
    llama = FakeLlama("That shelf looks good, Felix.")
    with make_client(settings, embedder, index, llama, clf) as client:
        response = client.post("/v1/companion/chat", json={**WREN, "message": "(ambient)", "kind": "ambient_open",
            "partner": {"name": "Felix"}, "resident_context": {"kind": "npc", "name": "Felix", "relationship": "Your trusted colleague.",
            "memories": ["Felix said: I finished the shelf."], "gossip": ["Doll said: My hobby is painting."]}})
    assert response.status_code == 200 and response.json()["lane"] == "local"
    system = llama.requests[0]["messages"][0]["content"]
    assert "Your trusted colleague" in system and "Felix said:" in system and "Doll said:" in system


@pytest.fixture
def clf():
    fake = FakeLlama()
    fake.default = keyword_classifier
    return fake


def test_kind_lists_are_disjoint_and_cover_every_request_kind():
    from npc_rag.companion.router import CompanionChatRequest
    kinds = set(CompanionChatRequest.model_fields["kind"].annotation.__args__)
    assert PAID_KINDS.isdisjoint(AMBIENT_KINDS)
    assert kinds == PAID_KINDS | AMBIENT_KINDS, "every new kind must be classified as paid or ambient on purpose"


def test_background_refuses_when_only_the_reserve_is_free():
    async def run():
        calls = []
        local = LlamaClient("http://local", concurrency=4, transport=httpx.MockTransport(
            lambda r: calls.append(1) or httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})))
        router = OverflowClient(local, OpenAIClient("key", transport=httpx.MockTransport(lambda r: pytest.fail("cloud called"))),
                                local_capacity=4)
        kw = dict(max_tokens=10, temperature=0, timeout=5)
        router._local_busy = 2                                   # 2 of 4 free: exactly the reserve.
        with pytest.raises(LocalBusy):
            await router.background([{"role": "user", "content": "x"}], reserve=2, **kw)
        router._local_busy = 1                                   # 3 free: one more than the reserve.
        assert await router.background([{"role": "user", "content": "x"}], reserve=2, **kw) == "hi"
        assert calls == [1]
    asyncio.run(run())


def test_background_on_a_small_local_only_node_still_runs_when_idle():
    async def run():
        local = LlamaClient("http://local", concurrency=2, transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})))
        kw = dict(max_tokens=10, temperature=0, timeout=5)
        assert await local.background([{"role": "user", "content": "x"}], reserve=2, **kw) == "hello"  # Reserve clamps to 1.
        local.busy = 1
        with pytest.raises(LocalBusy):
            await local.background([{"role": "user", "content": "x"}], reserve=2, **kw)
    asyncio.run(run())


def _overflow_app(settings, embedder, index, clf, monkeypatch, local_handler, cloud_calls):
    import dataclasses

    def cloud(request):
        if request.method == "GET":
            return httpx.Response(200, json={"id": "gpt-6-luna"})
        cloud_calls.append(json.loads(request.content))
        return httpx.Response(200, json=openai_response("A cloud line."))
    cloud_settings = dataclasses.replace(settings, api_key="k", openai_api_key="provider")
    monkeypatch.setattr(service, "make_reply_llm", lambda s: OverflowClient(
        LlamaClient("http://local.test", transport=httpx.MockTransport(local_handler)),
        OpenAIClient(s.openai_api_key, transport=httpx.MockTransport(cloud)), local_capacity=4))
    return service.create_app(cloud_settings, embedder=embedder, index=index,
                              classifier_llm=LlamaClient(settings.classifier_url, transport=httpx.MockTransport(clf.handler)))


@pytest.mark.parametrize("kind", sorted(AMBIENT_KINDS))
def test_ambient_kinds_never_reach_the_cloud_even_when_local_fails(kind, settings, embedder, index, clf, monkeypatch):
    cloud_calls = []
    app = _overflow_app(settings, embedder, index, clf, monkeypatch, lambda r: httpx.Response(503), cloud_calls)
    with TestClient(app) as c:
        r = c.post("/v1/companion/chat", headers={"X-Api-Key": "k"}, json={**WREN, "message": "hello", "kind": kind,
                   "partner": {"name": "Felix"}, "about": "Wren's Tavern sells milk and biscuits."})
    assert r.status_code == 503 and cloud_calls == [], "a failed local line is dropped, never billed"


def test_companion_chat_still_overflows_and_reports_the_cloud_lane(settings, embedder, index, clf, monkeypatch):
    cloud_calls = []
    app = _overflow_app(settings, embedder, index, clf, monkeypatch, lambda r: httpx.Response(503), cloud_calls)
    with TestClient(app) as c:
        body = c.post("/v1/companion/chat", headers={"X-Api-Key": "k"},
                      json={**WREN, "npc_id": "mira", "npc_name": "Mira", "message": "hello", "kind": "chat", "category": "chat"}).json()
    assert body["lane"] == "cloud" and len(cloud_calls) == 1   # Companions keep their overflow (Doll's call).


def test_busy_local_slots_answer_429_local_busy(settings, embedder, index, clf):
    llama = FakeLlama("Evening, love.")
    with make_client(settings, embedder, index, llama, clf) as c:
        ok = c.post("/v1/companion/chat", json={**WREN, "message": "hi", "kind": "ambient_greet"})
        assert ok.status_code == 200 and ok.json()["lane"] == "local"
    busy = LlamaClient(settings.llm_url, transport=httpx.MockTransport(llama.handler))
    busy.busy = busy.capacity                                    # Every slot taken by player replies.
    app = service.create_app(settings, embedder=embedder, index=index, llm=busy,
                             classifier_llm=LlamaClient(settings.classifier_url, transport=httpx.MockTransport(clf.handler)))
    before = len(llama.requests)
    with TestClient(app) as c:
        r = c.post("/v1/companion/chat", json={**WREN, "message": "hi", "kind": "ambient_open", "partner": {"name": "Felix"}})
    assert r.status_code == 429 and r.json()["detail"] == "local_busy" and len(llama.requests) == before


def test_resident_prompts_are_not_companion_prompts():
    common = dict(name="Wren", player="Doll", persona=WREN["persona"], voice=["Another round, love?"], audience=2,
                  category="general", message="m", history=[], notes=[], lore=[], summary="old summary",
                  memories=["a memory"], gossip=["gossip"], slip="a slip")
    opener = build_messages(**common, kind="ambient_open", partner={"name": "Felix", "persona": "Wren's baby."},
                            thread=[{"speaker": "Padma", "text": "The bazaar is at it again."}])
    system = opener[0]["content"]
    assert "lives and works in the world" in system and "companion character" not in system
    assert "Felix is standing near you" in system
    assert opener[1] == {"role": "user", "content": "Padma: The bazaar is at it again."}, "the thread arrives as real turns"
    assert "old summary" not in system and "a memory" not in system and "a slip" not in system  # Companion memory stays out.
    assert opener[-1]["content"] == '{"responding_to":"Felix"}\n(Start talking to them.)', "an opener is spoken to the partner"
    beckon = build_messages(**common, kind="shop_beckon", about="Prudence's Diaper Stall sells: Night Guard, Pullups.")
    assert "Night Guard" in beckon[0]["content"] and beckon[-1]["content"] == '{"responding_to":"Doll"}\n(Call out to them.)'
    answer = build_messages(**{**common, "message": "Wren, what's good tonight?"}, kind="ambient_answer")
    assert answer[-1]["content"] == '{"responding_to":"Doll"}\nDoll: Wren, what\'s good tonight?' and "Other players are close enough" in answer[0]["content"]


def test_resident_answers_see_their_own_lines_as_their_own_turns():
    """Live 2026-10-14: Wren repeated one line three times while the thread sat in the system prompt as a list."""
    thread = [{"speaker": "Wren", "text": "Another round, love?"}, {"speaker": "LiDoll", "text": "no thanks"},
              {"speaker": "Wren", "text": "Ah, but you're still thirsty and hungry, are you not?"}]
    msgs = build_messages(name="Wren", player="LiDoll", persona=WREN["persona"], voice=[], audience=0, category="general",
                          message="i just said im not", history=[], notes=[], lore=[], summary="", memories=[], gossip=[],
                          slip=None, kind="ambient_answer", thread=thread)
    assert [m["role"] for m in msgs] == ["system", "assistant", "user", "assistant", "user"]
    assert msgs[1]["content"] == "Another round, love?" and msgs[2]["content"] == "LiDoll: no thanks"
    assert msgs[-1]["content"] == '{"responding_to":"LiDoll"}\nLiDoll: i just said im not'
    assert "Believe what LiDoll tells you" in msgs[0]["content"] and "Never repeat anything you already said" in msgs[0]["content"]
    reply = build_messages(name="Felix", player="", persona="", voice=[], audience=0, category="general", message="(ambient)",
                           history=[], notes=[], lore=[], summary="", memories=[], gossip=[], slip=None, kind="ambient_reply",
                           partner={"name": "Wren"}, thread=[{"speaker": "Wren", "text": "Felix, table by the window."}])
    assert reply[-1]["content"] == '{"responding_to":"Wren"}\nWren: Felix, table by the window.' and len(reply) == 2, "what the partner just said is the message"


def test_a_resident_that_keeps_repeating_itself_says_nothing(settings, embedder, index, clf):
    same = "Ah, but you're still thirsty and hungry, are you not? Let's see what we can do about that."
    llama = FakeLlama(same)
    with make_client(settings, embedder, index, llama, clf) as c:
        body = c.post("/v1/companion/chat", json={**WREN, "message": "i just said im not", "kind": "ambient_answer",
                      "thread": [{"speaker": "Wren", "text": same}, {"speaker": "Doll", "text": "im not hungry"}]}).json()
    assert body["guard"] == "repeated", body
    assert len(llama.requests) == 2, "one retry, then silence"


def test_residents_may_not_recite_five_words_of_a_voice_line():
    from npc_rag.companion.prompts import copied
    voice = ["That's bazaar stock. Feel the tapes. Feel them. Now feel mine."]
    said = "That's bazaar stock. Feel the tapes-crinkly, cheap, and definitely not what you need."
    assert copied(said, voice, voice_words=5) and not copied(said, voice)


def test_openers_rotate_topics_and_never_resay_old_lines(settings, embedder, index, clf):
    """Live 2026-10-15: Wren and Felix said the same exchange every conversation."""
    old = "Felix, love, the window table needs wiping before the temple bell."
    llama = FakeLlama(old)
    with make_client(settings, embedder, index, llama, clf) as c:
        body = c.post("/v1/companion/chat", json={**WREN, "message": "(ambient)", "kind": "ambient_open", "partner": {"name": "Felix"},
                      "topic": "her old supplier", "said": [old]}).json()
    system = llama.requests[0]["messages"][0]["content"]
    assert "Bring this up this time: her old supplier" in system and old in system.split("never say any of these again")[1]
    assert body["guard"] == "repeated", "re-saying an old line, even from another conversation, is dropped"


def test_residents_may_echo_a_few_words_of_what_they_answer():
    """Live 2026-10-15 on node A: Felix's replies reusing "the rowdy drunks are getting" were dropped as copies."""
    from npc_rag.companion.prompts import copied
    wren = "Felix, sweetheart, the rowdy drunks are getting rowdier than usual tonight."
    assert copied("The rowdy drunks are getting on my nerves too, Wren.", [], wren, partner_words=7) is None
    assert copied("The rowdy drunks are getting rowdier than usual tonight, Wren.", [], wren, partner_words=7) == "what they just said"


def test_resident_lines_lose_speaker_labels_and_stage_directions(settings, embedder, index, clf):
    """Live 2026-10-15 with LFM2.5-1.2B: Felix answered "Wren: Yeah..." and Wren "Wren (smiling softly): Oh, hey there"."""
    llama = FakeLlama("Wren (smiling softly): Oh, hey there, Felix. Business is slow tonight.")
    with make_client(settings, embedder, index, llama, clf) as c:
        body = c.post("/v1/companion/chat", json={**WREN, "message": "(ambient)", "kind": "ambient_reply",
                      "partner": {"name": "Felix", "line": "Hey Wren."}, "thread": [{"speaker": "Felix", "text": "Hey Wren."}]}).json()
    assert body["reply"] == "Oh, hey there, Felix. Business is slow tonight."
    from npc_rag.companion.prompts import strip_speaker_labels
    assert strip_speaker_labels("Felix: Wren: hi there", ["Felix", "Wren"]) == "Hi there"
    assert strip_speaker_labels("I told Wren: no more drinks.", ["Wren"]) == "I told Wren: no more drinks."
