"""The companion pipeline (/v1/companion/*): lore gates, memories, gossip, slips, guard and failover signals.

Same harness as test_services: a real fixture wiki index, the hashing embedder and both model servers mocked.
"""
import dataclasses

import pytest

from test_services import FakeLlama, keyword_classifier, make_client  # noqa: F401 (fixtures below reuse them)

ELOWEN = {"npc_id": "paladin_aegis", "npc_name": "Elowen", "player_name": "Doll",
          "persona": "A patient, brave shield-bearer who guards her companions with a steady voice.",
          "voice": ["Stay behind me when things get rough. I can hold the line."]}

LORE = {"version": "v1", "sections": [
    {"npc": "paladin_aegis", "id": "elowen-home", "heading": "Monastery childhood",
     "text": "Elowen grew up in the shield monastery of the north and trained under Brother Aldric."},
    {"npc": "paladin_aegis", "id": "elowen-vow", "heading": "The broken vow", "reveal_bond": 20,
     "text": "Elowen once broke her shield vow and fled a battle; she has guarded others ever since to make up for it."},
    {"npc": "merchant_mira", "id": "mira-wall", "heading": "Name in stone", "reveal_flag": "story_mira_name_found",
     "text": "Mira carved her name in a dungeon wall when she escaped the Chrysalis."},
]}


@pytest.fixture
def loose(settings):
    return dataclasses.replace(settings, companion_lore_min=0.05, companion_memory_min=0.05, companion_gossip_min=0.05)


@pytest.fixture
def llama():
    return FakeLlama("Stay behind me, Doll.")


@pytest.fixture
def clf():
    fake = FakeLlama()
    fake.default = keyword_classifier
    return fake


def ask(c, **extra):
    return c.post("/v1/companion/chat", json={**ELOWEN, "message": "hello Elowen", **extra})


def system_of(llama, i=0):
    return llama.requests[i]["messages"][0]["content"]


def test_lore_push_status_and_version_check(loose, embedder, index, llama, clf):
    with make_client(loose, embedder, index, llama, clf) as c:
        assert c.get("/v1/companion/lore").json() == {"version": None, "sections": 0}
        assert ask(c, lore_version="v1").status_code == 409            # Fresh node: the game server must push first.
        assert c.post("/v1/companion/lore", json=LORE).json() == {"ok": True, "version": "v1", "sections": 3}
        assert ask(c, lore_version="v1").status_code == 200
        assert ask(c, lore_version="v2").status_code == 409            # Stale lore is refused, never silently used.
        assert ask(c).status_code == 200                                # No version sent: answer with whatever is loaded.


def test_secrets_unlock_with_bond_and_flags(loose, embedder, index, llama, clf):
    with make_client(loose, embedder, index, llama, clf) as c:
        c.post("/v1/companion/lore", json=LORE)
        low = ask(c, message="tell me about your shield vow and the battle", bond=5).json()
        assert "elowen-vow" not in low["lore"] and "broke her shield vow" not in system_of(llama, 0)
        high = ask(c, message="tell me about your shield vow and the battle", bond=25).json()
        assert "elowen-vow" in high["lore"] and "broke her shield vow" in system_of(llama, 1)
        mira = {**ELOWEN, "npc_id": "merchant_mira", "npc_name": "Mira"}
        assert c.post("/v1/companion/chat", json={**mira, "message": "where did you carve your name"}).json()["lore"] == []
        found = c.post("/v1/companion/chat", json={**mira, "message": "where did you carve your name", "flags": ["story_mira_name_found"]}).json()
        assert found["lore"] == ["mira-wall"]
        assert "Elowen" not in system_of(llama, 3).split("Your own past")[-1].split("How to reply")[0]  # Only Mira's own lore.


def test_persona_voice_and_lore_names_pass_the_guard(loose, embedder, index, llama, clf):
    llama.replies = ["Stay behind me when things get rough. I can hold the line. Brother Aldric taught me that."]
    with make_client(loose, embedder, index, llama, clf) as c:
        c.post("/v1/companion/lore", json=LORE)
        body = ask(c, message="who trained you in the monastery").json()
    system = system_of(llama)
    assert "You are Elowen" in system and ELOWEN["persona"] in system and "Stay behind me" in system
    assert body["guard"] == "passed"     # Copying a voice example is not a leak, and Aldric comes from the lore.
    assert body["category"] == "general" and body["method"] == "llm"


def test_invented_names_are_still_caught(loose, embedder, index, llama, clf):
    llama.replies = ["Ask Lord Zarvok, he knows.", "Lord Zarvok again."]
    with make_client(loose, embedder, index, llama, clf) as c:
        body = ask(c, message="who should I talk to").json()
    assert body["guard"] == "unsupported_terms" and body["reply"].startswith("I'm not sure")


def test_memories_rank_by_meaning_and_embarrassing_ones_wait_for_privacy(loose, embedder, index, llama, clf):
    memories = [
        {"id": "m-wyrm", "text": "Doll and I beat the Ember Wyrm at the caldera together.", "importance": 4, "age_days": 2},
        {"id": "m-bread", "text": "Doll bought sweet bread from the baker.", "importance": 1, "age_days": 40},
        {"id": "m-leak", "text": "Doll had an accident at the caldera hot springs and I helped her change.", "importance": 3,
         "age_days": 1, "embarrassing": True},
    ]
    with make_client(dataclasses.replace(loose, companion_memory_k=2), embedder, index, llama, clf) as c:
        public = ask(c, message="remember the caldera and the wyrm", memories=memories, audience=2).json()
        alone = ask(c, message="remember the caldera and the wyrm", memories=memories, audience=0).json()
    assert public["memories_offered"][0] == "m-wyrm" and "m-leak" not in public["memories_offered"]
    assert "hot springs" not in system_of(llama, 0) and "Other players are close enough" in system_of(llama, 0)
    assert "m-leak" in alone["memories_offered"] and "hot springs" in system_of(llama, 1)
    assert "Nobody else is close enough" in system_of(llama, 1)


def test_gossip_fills_a_silence_but_never_beats_a_shared_memory(loose, embedder, index, llama, clf):
    gossip = [{"id": "g1", "text": "Someone I traveled with got lost in the frozen tundra for a whole night."}]
    with make_client(loose, embedder, index, llama, clf) as c:
        lonely = ask(c, message="any stories about the frozen tundra", gossip=gossip).json()
        own = ask(c, message="any stories about the frozen tundra", gossip=gossip,
                  memories=[{"id": "m1", "text": "Doll and I crossed the frozen tundra in a storm."}]).json()
    assert lonely["gossip"] == "g1" and "Never name or describe who" in system_of(llama, 0)
    assert own["gossip"] is None and own["memories_offered"] == ["m1"]


def test_slips_need_listeners_and_must_really_be_said(loose, embedder, index, llama, clf):
    slip = {"id": "m-leak", "text": "Doll wet her diaper at the caldera hot springs.", "embarrassing": True}
    llama.replies = ["Careful near the hot springs, Doll, we both remember the caldera diaper incident!",
                     "Lovely weather today, Doll.", "Nice to see you."]
    with make_client(loose, embedder, index, llama, clf) as c:
        said = ask(c, slip=slip, audience=3).json()
        quiet = ask(c, slip=slip, audience=3).json()
        alone = ask(c, slip=slip, audience=0).json()
    assert said["slip"] == {"memory_id": "m-leak"} and "can't resist teasing Doll" in system_of(llama, 0)
    assert quiet["slip"] is None, "offered but not said: no Dignity cost"
    assert alone["slip"] is None and "can't resist teasing" not in system_of(llama, 2), "nobody listening: never offered"


def test_game_questions_use_the_wiki_and_caller_category_skips_the_classifier(loose, embedder, index, llama, clf):
    llama.replies = ["Visit the Cursebreaker, Doll."]
    with make_client(loose, embedder, index, llama, clf) as c:
        game = ask(c, message="how do I remove a cursed item").json()
        assert "Guidebook notes" in system_of(llama, 0) and game["sources"] and game["category"] == "game"
        n = len(clf.requests)
        skipped = ask(c, message="how do I remove a cursed item", category="game").json()
    assert len(clf.requests) == n and skipped["method"] == "caller"


def test_game_question_with_nothing_to_go_on_skips_the_model(settings, embedder, index, llama, clf):
    strict = dataclasses.replace(settings, min_relevance=0.99, companion_lore_min=0.99, companion_memory_min=0.99)
    with make_client(strict, embedder, index, llama, clf) as c:
        body = ask(c, message="can I get a pet dragon").json()
    assert body["guard"] == "no_relevant_notes" and llama.requests == []


def test_model_down_answers_503_so_the_game_server_fails_over(loose, embedder, index, llama, clf):
    llama.replies = [500, 500, 500]
    with make_client(loose, embedder, index, llama, clf) as c:
        assert ask(c).status_code == 503


def test_validation(loose, embedder, index, llama, clf):
    with make_client(loose, embedder, index, llama, clf) as c:
        assert ask(c, bond=101).status_code == 422
        assert ask(c, memories=[{"id": "x", "text": "y" * 401}]).status_code == 422
        assert c.post("/v1/companion/chat", json={"message": "hi"}).status_code == 422          # npc_id and npc_name required.
        assert c.post("/v1/companion/lore", json={"version": "v", "sections": [{"npc": "a", "id": "b", "heading": "c", "text": ""}]}).status_code == 422


def test_impressions_are_framed_as_what_the_player_said(loose, embedder, index, llama, clf):
    llama.replies = ['Sure: {"facts": [{"text": "Doll said she loves strawberry milk.", "importance": 2},'
                     ' {"text": "she is afraid of the dark", "importance": 9}, {"text": "x"}, "Doll said hi there friend",'
                     ' {"text": "Doll said four"}]}', "no json at all", 500, 500, 500]
    with make_client(loose, embedder, index, llama, clf) as c:
        body = c.post("/v1/companion/impressions", json={"npc_name": "Elowen", "player_name": "Doll",
                                                         "message": "I love strawberry milk and I'm scared of the dark", "reply": "Noted."}).json()
        assert body["facts"] == [{"text": "Doll said she loves strawberry milk.", "importance": 2, "embarrassing": False},
                                 {"text": "Doll said: she is afraid of the dark", "importance": 3, "embarrassing": False},
                                 {"text": "Doll said hi there friend", "importance": 1, "embarrassing": False}]  # Capped at 3, clamped, always "Doll said".
        assert "Output JSON only" in llama.requests[0]["messages"][0]["content"] and llama.requests[0]["temperature"] == 0.0
        assert c.post("/v1/companion/impressions", json={"npc_name": "Elowen", "player_name": "Doll", "message": "hi"}).json() == {"facts": []}
        assert c.post("/v1/companion/impressions", json={"npc_name": "Elowen", "player_name": "Doll", "message": "hi"}).status_code == 503


def test_summaries_merge_memories_in_first_person(loose, embedder, index, llama, clf):
    llama.replies = ["**We beat the Ember Wyrm together and Doll trusts me.** I like traveling with her.", 500, 500, 500]
    with make_client(loose, embedder, index, llama, clf) as c:
        body = c.post("/v1/companion/summarize", json={"npc_name": "Elowen", "player_name": "Doll", "summary": "We just met.",
                                                       "memories": ["Doll and I beat the Ember Wyrm together.", "Doll said she loves strawberry milk."]}).json()
        assert body == {"summary": "We beat the Ember Wyrm together and Doll trusts me. I like traveling with her."}
        user = llama.requests[0]["messages"][1]["content"]
        assert "Previous note: We just met." in user and "- Doll said she loves strawberry milk." in user
        assert c.post("/v1/companion/summarize", json={"npc_name": "Elowen", "player_name": "Doll", "memories": ["x"]}).status_code == 503
        assert c.post("/v1/companion/summarize", json={"npc_name": "Elowen", "player_name": "Doll", "memories": []}).status_code == 422


def test_slip_detection_needs_an_embarrassing_word_and_a_shared_detail():
    """Live run 2026-10-12: "you slipped on the moss in the Forest" was wrongly counted as an accident slip."""
    from npc_rag.companion.prompts import mentions
    memory, names = "Doll had an accident in the Forest while we were traveling together.", {"doll", "dame", "elowen"}
    assert not mentions("Doll, you were the one who slipped on the moss last time in the Forest.", memory, names)
    assert not mentions("The Forest is lovely this time of year, we were there together.", memory, names)
    assert not mentions("Did you remember to change your diaper? Let's head home.", memory, names)  # Embarrassing, but not this memory.
    assert mentions("Remember your little accident in the Forest, Doll?", memory, names)
    assert mentions("You had a little mishap in the Forest, but I made sure no one saw it.", memory, names)


def test_summary_prompt_keeps_embarrassing_moments_out_and_avoids_pronouns():
    from npc_rag.companion.reflect import summary_messages
    system = summary_messages(npc="Elowen", player="Doll", summary="", memories=["x"])[0]["content"]
    assert "Leave out accidents" in system and "may be read in public" in system and "don't guess their pronouns" in system


def test_embarrassing_impressions_are_flagged():
    from npc_rag.companion.reflect import parse_impressions
    facts = parse_impressions('{"facts": [{"text": "Doll said they need help changing their diaper."}, {"text": "Doll said they love tea."}]}', "Doll")
    assert [f["embarrassing"] for f in facts] == [True, False]


def test_slips_must_be_about_the_player_not_the_companion():
    """Live 2026-10-12 on both nodes: the model flipped whose accident it was ("I had an accident there once")."""
    from npc_rag.companion.prompts import mentions
    memory, names = "Doll had an accident in the Forest while we were traveling together.", {"doll", "dame", "elowen"}
    assert not mentions("The Forest is tricky. Doll, you were there when I had that little accident, weren't you?", memory, names, "Doll")
    assert not mentions("The Forest is beautiful. I had an accident there once, but Doll helped me up.", memory, names, "Doll")
    assert mentions("Doll had an accident in the Forest, and I kept watch.", memory, names, "Doll")
    assert mentions("Remember your little accident in the Forest?", memory, names, "Doll")


def test_summaries_drop_embarrassing_sentences_in_code():
    from npc_rag.companion.reflect import tidy_summary
    text = tidy_summary("We beat the Slime Knight, even after the accident. Doll told me she loves strawberry milk. We are close.")
    assert text == "Doll told me she loves strawberry milk. We are close."
