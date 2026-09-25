"""Both services end to end: the agent talks to the real RAG app in-process; only llama.cpp is mocked."""
import dataclasses
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from npc_rag import agent_service, rag_service
from npc_rag.llm import LlamaClient
from npc_rag.prompts import tidy_reply


class FakeLlama:
    """Records every request and answers with a queued reply (or an HTTP error)."""

    def __init__(self):
        self.requests: list[dict] = []
        self.replies: list = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content)
        self.requests.append(body)
        reply = self.replies.pop(0) if self.replies else "Hello from the guide!"
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": "boom"})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": reply}}]})


@pytest.fixture
def llama():
    return FakeLlama()


def make_client(settings, embedder, index, llama):
    rag_app = rag_service.create_app(settings, embedder=embedder, index=index)
    llm = LlamaClient(settings.llm_url, transport=httpx.MockTransport(llama.handler))
    app = agent_service.create_app(settings, embedder=embedder, llm=llm,
                                   rag_transport=httpx.ASGITransport(app=rag_app))
    return TestClient(app)


def test_rag_search_endpoint(settings, embedder, index):
    with TestClient(rag_service.create_app(settings, embedder=embedder, index=index)) as c:
        r = c.post("/v1/search", json={"query": "cursebreaker cursed item", "k": 2})
        assert r.status_code == 200
        body = r.json()
        assert body["results"][0]["id"] == "needs#cursed-equipment:0"
        assert body["results"][0]["url"] == "https://wiki.test/wiki/#/needs/cursed-equipment"
        assert body["relevant"] is True


def test_rag_reindex_from_local_folder(settings, embedder, index):
    with TestClient(rag_service.create_app(settings, embedder=embedder, index=index)) as c:
        r = c.post("/v1/reindex", json={})
        assert r.status_code == 200 and r.json()["index"]["pages"] == 2
        assert (settings.index_dir / "vectors.npy").is_file()
        bad = c.post("/v1/reindex", json={"source": str(settings.index_dir / "nowhere")})
        assert bad.status_code == 502  # Old index kept.
        assert c.post("/v1/search", json={"query": "toilet"}).status_code == 200


def test_game_question_gets_wiki_notes(settings, embedder, index, llama):
    llama.replies = ["**Pip:** Visit the Cursebreaker in the Market Hall! *waves*"]
    with make_client(settings, embedder, index, llama) as c:
        r = c.post("/v1/npc/chat", json={"message": "how do I remove a cursed item", "player_name": "Doll"})
    body = r.json()
    assert r.status_code == 200 and body["category"] == "game" and not body["fallback"]
    assert body["reply"] == "Visit the Cursebreaker in the Market Hall!"  # Markdown, name prefix and stage direction removed.
    system = llama.requests[0]["messages"][0]["content"]
    assert "Guidebook notes" in system and "Cursebreaker" in system and "Doll" in system
    assert llama.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["sources"] and body["sources"][0]["url"].startswith("https://wiki.test/wiki/#/needs/")


def test_general_chat_has_no_notes(settings, embedder, index, llama):
    with make_client(settings, embedder, index, llama) as c:
        body = c.post("/v1/npc/chat", json={"message": "hello there friend"}).json()
    assert body["category"] == "general" and body["sources"] == []
    assert "Guidebook notes" not in llama.requests[0]["messages"][0]["content"]


def test_memory_follows_the_player(settings, embedder, index, llama):
    llama.replies = ["First answer.", "Second answer."]
    with make_client(settings, embedder, index, llama) as c:
        c.post("/v1/npc/chat", json={"message": "where is the toilet", "player_id": "p1"})
        c.post("/v1/npc/chat", json={"message": "hello there friend", "player_id": "p1"})
        second = llama.requests[1]["messages"]
        assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
        assert second[2]["content"] == "First answer."
        c.delete("/v1/npc/session/p1")
        c.post("/v1/npc/chat", json={"message": "hello there friend", "player_id": "p1"})
        assert len(llama.requests[2]["messages"]) == 2  # Forgotten.


def test_caller_history_replaces_memory(settings, embedder, index, llama):
    with make_client(settings, embedder, index, llama) as c:
        c.post("/v1/npc/chat", json={"message": "hello there friend", "player_id": "p2",
                                     "history": [{"role": "player", "content": "hi"}, {"role": "npc", "content": "hey!"}]})
    assert [m["role"] for m in llama.requests[0]["messages"]] == ["system", "user", "assistant", "user"]


def test_llm_failure_uses_fallback_line(settings, embedder, index, llama):
    llama.replies = [500]
    with make_client(settings, embedder, index, llama) as c:
        body = c.post("/v1/npc/chat", json={"message": "where is the toilet"}).json()
    assert body["fallback"] is True and "wiki" in body["reply"]


def test_thinking_blocks_are_removed(settings, embedder, index, llama):
    llama.replies = ["<think>the player wants a joke</think>Why did the slime cross the road?"]
    with make_client(settings, embedder, index, llama) as c:
        body = c.post("/v1/npc/chat", json={"message": "tell me a joke please"}).json()
    assert body["reply"] == "Why did the slime cross the road?"


def test_validation_and_api_key(settings, embedder, index, llama):
    with make_client(settings, embedder, index, llama) as c:
        assert c.post("/v1/npc/chat", json={"message": "x" * 201}).status_code == 422
        assert c.post("/v1/npc/chat", json={"message": "   "}).status_code == 422
    locked = dataclasses.replace(settings, api_key="s3cret")
    with make_client(locked, embedder, index, llama) as c:
        assert c.post("/v1/npc/chat", json={"message": "hi"}).status_code == 401
        ok = c.post("/v1/npc/chat", json={"message": "hello there friend"}, headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        assert c.get("/health").json()["rag"] is True  # Health stays open; RAG accepted the forwarded key.


def test_tidy_reply_cuts_at_sentence():
    text = "One sentence here. " * 30
    out = tidy_reply(text, "Pip", 100)
    assert len(out) <= 100 and out.endswith(".")
