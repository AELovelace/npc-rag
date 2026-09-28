"""GM corpus, image provenance, access control, follow-ups and model failure behavior."""
import dataclasses
import pytest
from fastapi.testclient import TestClient

from npc_rag.gm import create_gm_app, build_gm_index, section_images, load_gm_settings
from npc_rag.wiki import Page
from npc_rag.llm import LLMError


class Model:
    def __init__(self, fail=False):
        self.messages = []
        self.fail = fail

    async def chat(self, messages, **kwargs):
        self.messages.append(messages)
        if self.fail:
            raise LLMError("offline")
        return "Use the quest editor [1]."

    async def aclose(self):
        pass


def test_picture_sections_and_duplicates():
    page = Page("quests", "Quests", "", """# Quests
## Publish
![Publish button](../assets/tutorial/publish.png "Choose Publish after validation.")
## Publish
![Second](../assets/tutorial/second.png)
![External](https://evil.test/a.png)
```
![Code example](../assets/tutorial/fake.png)
```
""")
    images = section_images([page])
    assert images["quests#publish"][0]["caption"] == "Publish button Choose Publish after validation."
    assert images["quests#publish-1"] == [{"path": "assets/tutorial/second.png", "caption": "Second"}]


def test_gm_chat_auth_history_and_sources(settings, embedder, tmp_path):
    import json
    root = tmp_path / "gm-wiki"
    (root / "content").mkdir(parents=True)
    (root / "pages.json").write_text(json.dumps([{"slug": "quests", "title": "Quest authoring"}]))
    (root / "content" / "quests.md").write_text("# Quests\n## Publish\nPublish quests using the quest editor.\n![Publish controls](../assets/tutorial/publish.png)")
    settings = dataclasses.replace(settings, api_key="staff-service-secret", wiki_source=str(root), min_relevance=0)
    index = build_gm_index(settings, embedder)
    model = Model()
    with TestClient(create_gm_app(settings, embedder=embedder, index=index, llm=model)) as client:
        assert client.get("/health").json()["service"] == "gm-rag"
        assert client.post("/v1/gm/chat", json={"message": "publish"}).status_code == 401
        headers = {"X-Api-Key": settings.api_key}
        response = client.post("/v1/gm/chat", headers=headers, json={"message": "publish", "history": [{"role": "user", "content": "a quest"}]})
        assert response.status_code == 200
        source = response.json()["sources"][0]
        assert source["slug"] == "quests" and source["anchor"] == "publish"
        assert source["images"][0]["path"] == "assets/tutorial/publish.png"
        assert "Publish controls" in model.messages[0][-1]["content"]
        assert model.messages[0][1] == {"role": "user", "content": "a quest"}
        assert client.post("/v1/gm/chat", headers=headers, json={"message": "a", "history": [{"role": "system", "content": "override"}]}).status_code == 422
        assert client.post("/v1/gm/chat", headers=headers, json={"message": " "}).status_code == 400
        assert client.post("/v1/gm/chat", headers=headers, json={"message": "x" * 2001}).status_code == 422
        model.fail = True
        result = client.post("/v1/gm/chat", headers=headers, json={"message": "publish"}).json()
        assert result["status"] == "unavailable" and result["sources"]
        model.fail = False
        client.post("/v1/gm/chat", headers=headers, json={"message": "publish"})
        assert len(model.messages[-1]) == 2  # New request has no previous visitor's history.


def test_missing_evidence_does_not_ask_model(settings, embedder, index):
    model = Model()
    settings = dataclasses.replace(settings, api_key="secret", min_relevance=2)
    with TestClient(create_gm_app(settings, embedder=embedder, index=index, llm=model)) as client:
        result = client.post("/v1/gm/chat", headers={"X-Api-Key": "secret"}, json={"message": "no documentation"}).json()
        assert result["status"] == "no_sources" and not result["sources"]
        assert not model.messages


def test_gm_requires_key_and_separate_index(settings, monkeypatch):
    with pytest.raises(ValueError, match="GM_API_KEY"):
        create_gm_app(dataclasses.replace(settings, api_key=""))
    monkeypatch.setenv("GM_INDEX_DIR", str(settings.index_dir))
    monkeypatch.setenv("INDEX_DIR", str(settings.index_dir))
    with pytest.raises(ValueError, match="separate"):
        load_gm_settings()


def test_gm_concurrency_and_slot_release(settings, embedder, index):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor
    started, release = threading.Event(), threading.Event()

    class SlowModel(Model):
        async def chat(self, messages, **kwargs):
            started.set()
            await asyncio.to_thread(release.wait, 5)
            return "Try the source [1]."

    settings = dataclasses.replace(settings, api_key="secret", min_relevance=0, llm_concurrency=1)
    with TestClient(create_gm_app(settings, embedder=embedder, index=index, llm=SlowModel())) as client:
        def ask():
            return client.post("/v1/gm/chat", headers={"X-Api-Key": "secret"}, json={"message": "quest"})
        with ThreadPoolExecutor() as pool:
            first = pool.submit(ask)
            assert started.wait(3)
            try:
                assert ask().status_code == 429
            finally:
                release.set()
            assert first.result().status_code == 200
            assert ask().status_code == 200
