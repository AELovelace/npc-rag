"""Shared fixtures: a tiny wiki, the hashing embedder and settings that never touch the network."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from npc_rag.config import load_settings
from npc_rag.embedder import HashEmbedder
from npc_rag.index import WikiIndex
from npc_rag.wiki import load_pages

FIXTURE_WIKI = Path(__file__).parent / "fixtures" / "wiki"


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def settings(tmp_path):
    persona = tmp_path / "persona.md"
    persona.write_text("You are a cheerful test guide who loves helping new players find their way.", encoding="utf-8")
    return dataclasses.replace(
        load_settings(),
        llm_url="http://llama.test", classifier_url="http://classifier.test", api_key="", llm_api_key="",
        wiki_source=str(FIXTURE_WIKI), wiki_public_url="https://wiki.test/wiki/",
        index_dir=tmp_path / "index", persona_path=persona,
        rag_game_evidence=0.5, min_relevance=0.2, guard_keep_margin=1.0, guard_min_words=3, guard_retries=1, guard_allowed_terms=(),
        session_turns=6, max_message_chars=200, max_reply_chars=300,
    )


@pytest.fixture
def index(settings, embedder):
    return WikiIndex.build(load_pages(str(FIXTURE_WIKI)), embedder, settings.wiki_public_url, 60, str(FIXTURE_WIKI))
