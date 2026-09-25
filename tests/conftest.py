"""Shared fixtures: a tiny wiki, the hashing embedder and settings that never touch the network."""
from __future__ import annotations

import dataclasses
import json
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
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({
        "game": ["where is the toilet", "how do I remove a cursed item", "what does RPP do", "how do levels work"],
        "general": ["hello there friend", "how are you today", "tell me a joke please", "what is your favourite colour"],
    }), encoding="utf-8")
    persona = tmp_path / "persona.md"
    persona.write_text("You are a cheerful test guide.", encoding="utf-8")
    return dataclasses.replace(
        load_settings(),
        llm_url="http://llama.test", rag_url="http://rag.test", api_key="", llm_api_key="",
        wiki_source=str(FIXTURE_WIKI), wiki_public_url="https://wiki.test/wiki/",
        index_dir=tmp_path / "index", seeds_path=seeds, persona_path=persona,
        classifier_margin=0.05, rag_game_evidence=0.5, llm_tiebreak=False, min_relevance=0.2,
        session_turns=6, max_message_chars=200, max_reply_chars=300,
    )


@pytest.fixture
def index(settings, embedder):
    return WikiIndex.build(load_pages(str(FIXTURE_WIKI)), embedder, settings.wiki_public_url, 60, str(FIXTURE_WIKI))
