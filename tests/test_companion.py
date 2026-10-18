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
    return dataclasses.replace(settings, companion_lore_min=0.05, companion_locked_min=0.05, companion_memory_min=0.05, companion_gossip_min=0.05)


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
    assert body["guard"] in ("passed", "passed_after_retry")  # Copying a voice example is not a leak (the copy guard may retry), and Aldric comes from the lore.
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
                                 {"text": "Doll said: she is afraid of the dark", "importance": 3, "embarrassing": False}]  # Clamped, always "Doll said"; ungrounded facts dropped.
        assert "Output JSON only" in llama.requests[0]["messages"][0]["content"] and llama.requests[0]["temperature"] == 0.0
        assert '"\\"Doll said "' in llama.requests[0]["grammar"]  # Grammar-forced JSON with this player's prefix.
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


# ── Phase A (COMPANION_LIFELIKE_PLAN.md): cacheable prompts, kinds, situation, token cap, trimmed notes, model pool ──

def test_prompt_starts_with_a_static_block_shared_by_every_player(loose, embedder, index, llama, clf):
    with make_client(loose, embedder, index, llama, clf) as c:
        ask(c, message="hello Elowen", player_name="Doll", audience=0)
        ask(c, message="how do I remove a cursed item", player_name="Bea", audience=3)
    a, b = system_of(llama, 0), system_of(llama, 1)
    shared = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    assert "Keep everything friendly and non-sexual." in a[:shared] and a[:shared].endswith("with a player named ")  # Everything up to the name is identical.
    assert "named Doll" not in a[:shared] and "Right now you are with a player named Doll" in a  # ("Doll" alone is in "LiDollQuest".)


def test_companion_replies_use_the_companion_token_cap(loose, embedder, index, llama, clf):
    with make_client(dataclasses.replace(loose, companion_max_tokens=77), embedder, index, llama, clf) as c:
        ask(c)
    assert llama.requests[0]["max_tokens"] == 77


def test_companion_game_notes_are_fewer_and_shorter(loose, embedder, index, llama, clf):
    tight = dataclasses.replace(loose, companion_note_k=1, companion_note_words=5)
    with make_client(tight, embedder, index, llama, clf) as c:
        ask(c, message="how do I remove a cursed item", category="game")
    notes = system_of(llama).split("Guidebook notes:")[1]
    assert notes.count("[") == 1 and len(notes.split("\n", 2)[2].split()) <= 5


def test_kinds_change_the_mode_and_skip_the_classifier(loose, embedder, index, llama, clf):
    partner = {"name": "Wick", "persona": "A thoughtful former cultist.", "line": "Have you seen the desert?"}
    with make_client(loose, embedder, index, llama, clf) as c:
        emote = ask(c, message="*Doll hugs Elowen*", kind="emote").json()
        react = ask(c, message="won a boss fight", kind="reaction", about="We just beat the Ember Wyrm together.").json()
        muse = ask(c, message="idle", kind="musing", situation=["in Honeydew Village, a safe town", "it is evening"],
                   memories=[{"id": "m1", "text": "Doll and I beat the Slime Knight."}]).json()
        banter = ask(c, message="banter", kind="banter", partner=partner, audience=2,
                     memories=[{"id": "m1", "text": "Doll and I beat the Slime Knight."}],
                     slip={"id": "s", "text": "Doll had an accident.", "embarrassing": True}).json()
    assert clf.requests == [] and all(r["method"] == "kind" for r in (emote, react, muse, banter))
    assert "did something instead of speaking" in system_of(llama, 0)
    assert "What happened: We just beat the Ember Wyrm together." in system_of(llama, 1)
    assert llama.requests[1]["messages"][-1]["content"] == '{"responding_to":"Doll"}\n(React out loud.)'
    assert llama.requests[0]["messages"][-1]["content"] == '{"responding_to":"Doll"}\n*Doll hugs Elowen*', "an emote already names its actor"
    assert llama.requests[3]["messages"][-1]["content"] == '{"responding_to":"Wick"}\n(Speak to them.)', "banter answers the partner"
    assert "What you notice right now" in system_of(llama, 2) and "it is evening" in system_of(llama, 2)
    assert "say one short thing on your own" in system_of(llama, 2)
    banter_system = system_of(llama, 3)
    assert 'They just said to you: "Have you seen the desert?"' in banter_system and "Slime Knight" not in banter_system
    assert "teasing" not in banter_system and banter["slip"] is None, "banter never carries player memories or slips"


def test_situation_and_partner_are_validated(loose, embedder, index, llama, clf):
    with make_client(loose, embedder, index, llama, clf) as c:
        assert ask(c, situation=["x" * 161]).status_code == 422
        assert ask(c, situation=["fact"] * 11).status_code == 422
        assert ask(c, kind="gossiping").status_code == 422
        assert ask(c, kind="banter", partner={"name": ""}).status_code == 422


def test_llm_pool_spreads_over_copies_and_skips_a_dead_one():
    import asyncio
    import httpx
    from npc_rag.llm import LlamaClient, LlamaPool, make_llm
    seen = []

    def copy(name, fail=False):
        async def handler(request):
            await asyncio.sleep(0.02)                          # A real server takes time, so concurrent calls overlap.
            seen.append(name)
            if fail:
                return httpx.Response(500, json={})
            return httpx.Response(200, json={"choices": [{"message": {"content": "from " + name}}]})
        return LlamaClient("http://" + name, transport=httpx.MockTransport(handler))

    async def run():
        pool = LlamaPool([copy("a"), copy("b")])
        replies = await asyncio.gather(*(pool.chat([{"role": "user", "content": "hi"}], max_tokens=5, temperature=0, timeout=5) for _ in range(4)))
        assert {"from a", "from b"} <= set(replies), "concurrent calls use both copies"
        broken = LlamaPool([copy("dead", fail=True), copy("alive")])
        assert await broken.chat([{"role": "user", "content": "hi"}], max_tokens=5, temperature=0, timeout=5) == "from alive"
        await pool.aclose(); await broken.aclose()
    asyncio.run(run())
    assert isinstance(make_llm("http://127.0.0.1:9090"), LlamaClient)
    assert isinstance(make_llm("http://127.0.0.1:9090, http://127.0.0.1:9091"), LlamaPool)


def test_llm_pool_sends_more_work_to_the_faster_copy():
    """Node A 2026-10-18: one card ~4x slower than the other. Least-busy alone split replies evenly; the pool now weighs
    each copy by its recent reply time, so the fast copy takes most of a steady stream and the slow one still helps."""
    import asyncio
    import httpx
    from npc_rag.llm import LlamaClient, LlamaPool
    seen = []

    def copy(name, seconds):
        async def handler(request):
            await asyncio.sleep(seconds)
            seen.append(name)
            return httpx.Response(200, json={"choices": [{"message": {"content": name}}]})
        return LlamaClient("http://" + name, concurrency=4, transport=httpx.MockTransport(handler))

    async def run():
        pool = LlamaPool([copy("slow", 0.08), copy("fast", 0.02)])
        async def player():
            for _ in range(10):
                await pool.chat([{"role": "user", "content": "hi"}], max_tokens=5, temperature=0, timeout=5)
        await asyncio.gather(*(player() for _ in range(4)))
        view = pool.copies()
        await pool.aclose()
        return view
    view = asyncio.run(run())
    assert seen.count("fast") >= 2 * seen.count("slow") > 0, f"fast {seen.count('fast')} vs slow {seen.count('slow')}"
    assert view[1]["recent_s"] < view[0]["recent_s"] and all(v["in_flight"] == 0 for v in view)


def test_copied_lines_are_retried_once(loose, embedder, index, llama, clf):
    """Continuity test 2026-10-13 (Qwen3-4B): a hug got the voice line back; banter parroted the partner."""
    from npc_rag.companion.prompts import copied
    voice = ["Come here little one, let Matron check your padding~"]
    assert copied("Come here little one, let Matron check your padding~ How sweet.", voice) == "one of your example lines"
    assert copied("You're standing too close to the edge, Sable.", [], "You're standing too close to the edge, Wick.") == "what they just said"
    assert copied("A brand new thought.", voice, "Something else entirely.") is None
    llama.replies = ["Come here little one, let Matron check your padding~", "Oh! A hug? Come here, dear, Matron hugs back."]
    with make_client(loose, embedder, index, llama, clf) as c:
        body = c.post("/v1/companion/chat", json={**ELOWEN, "npc_id": "matron_haven", "npc_name": "Rosalind", "voice": voice,
                                                  "message": "*Doll hugs Rosalind*", "kind": "emote"}).json()
    assert body["reply"] == "Oh! A hug? Come here, dear, Matron hugs back." and body["guard"] == "passed_after_retry"
    assert "repeated one of your example lines" in llama.requests[1]["messages"][0]["content"]


def test_open_secrets_are_tagged_and_locked_ones_leave_only_a_marker(loose, embedder, index, llama, clf):
    with make_client(loose, embedder, index, llama, clf) as c:
        c.post("/v1/companion/lore", json=LORE)
        low = ask(c, message="tell me about your shield vow and the battle", bond=5).json()
        system = system_of(llama, 0)
        assert low["locked_secret"] is True and '{"secret":"locked"} Something you are not ready to tell Doll yet.' in system
        assert "broke her shield vow" not in system and "The broken vow" not in system   # Neither the text nor the heading leaks.
        assert "never say you have no past" in system                                       # The legend comes with the marker.
        high = ask(c, message="tell me about your shield vow and the battle", bond=25).json()
        system = system_of(llama, 1)
        assert high["locked_secret"] is False and '- {"secret":"open"} The broken vow: Elowen once broke' in system
        assert "- Monastery childhood: Elowen grew up" in system                          # Ungated lore stays untagged.


def test_prompts_without_secrets_are_byte_identical_to_before():
    from types import SimpleNamespace
    from npc_rag.companion.prompts import LORE, build_messages
    plain = SimpleNamespace(heading="Monastery childhood", text="Elowen grew up in the north.", reveal_bond=0, reveal_flag="")
    args = dict(name="Elowen", player="Doll", persona="A shield-bearer.", voice=[], audience=0, category="general", message="hi",
                history=[], notes=[], lore=[plain], summary="", memories=[], gossip=[], slip=None)
    system = build_messages(**args)[0]["content"]
    assert LORE.format(items="- Monastery childhood: Elowen grew up in the north.") in system and "secret" not in system
    assert "secret" not in build_messages(**{**args, "lore": []})[0]["content"]


def test_banter_never_shares_a_players_unlocked_secret():
    from types import SimpleNamespace
    from npc_rag.companion.prompts import build_messages
    secret = SimpleNamespace(heading="The broken vow", text="Elowen once fled a battle.", reveal_bond=20, reveal_flag="")
    system = build_messages(name="Elowen", player="", persona="A shield-bearer.", voice=[], audience=0, category="general",
                            message="banter", history=[], notes=[], lore=[secret], summary="", memories=[], gossip=[], slip=None,
                            kind="banter", partner={"name": "Mira", "persona": "A peddler."}, locked=True)[0]["content"]
    assert "fled a battle" not in system and "secret" not in system


def test_impressions_grammar_and_grounding():
    """lidoll-fm-0.2 (2026-10-17) answered impressions in prose; the grammar forces JSON, and grounding drops what it invents."""
    from npc_rag.companion.reflect import grounded, impressions_grammar, parse_impressions
    g = impressions_grammar('Kit "the" Brave')
    assert r'"\"Kit the Brave said "' in g and r'char ::= [^"\\\n\r]' in g  # Name sanitized; only quote, backslash, CR/LF banned (not the letter n).
    assert grounded("Kit said they love peach tea", "I love peach tea", "Kit")
    assert not grounded("Moss said how to open your inventory", "how do I open my inventory?", "Moss")  # A question, not a fact.
    assert not grounded("Moss said they collect rare stamps", "how do I open my inventory?", "Moss")  # Invented.
    raw = '{"facts": [{"text": "Kit said they love peach tea", "importance": 2}, {"text": "Kit said they own a castle", "importance": 3}]}'
    assert [f["text"] for f in parse_impressions(raw, "Kit", message="I love peach tea")] == ["Kit said they love peach tea"]
    assert len(parse_impressions(raw, "Kit")) == 2  # Without the message (older callers) nothing is filtered.


def test_the_final_turn_names_who_the_companion_answers(loose, embedder, index, clf):
    """lidoll-fm-0.2 lost track of the player (2026-10-17): the name sat only in the system prompt. Now every player turn
    carries the speaker and the last one starts with a responding_to marker; an echoed marker or self-label is stripped."""
    llama = FakeLlama('{"responding_to":"Doll"} Elowen: Stay behind me, Doll.')
    with make_client(loose, embedder, index, llama, clf) as c:
        body = ask(c, message="are you ready?", history=[{"role": "player", "content": "hi Elowen"},
                                                        {"role": "npc", "content": "Hello, Doll."}]).json()
    msgs = llama.requests[-1]["messages"]
    assert msgs[-1]["content"] == '{"responding_to":"Doll"}\nDoll: are you ready?'
    assert [m["content"] for m in msgs[1:-1]] == ["Doll: hi Elowen", "Hello, Doll."], "player history is labelled, own lines are not"
    assert body["reply"] == "Stay behind me, Doll."


def test_responding_to_escapes_names_and_follows_the_last_speaker():
    from npc_rag.companion.prompts import address, addressee, strip_markers
    assert address("hi", 'Bo "Tiny" Lee') == '{"responding_to":"Bo \\"Tiny\\" Lee"}\nhi'
    thread = [{"speaker": "Doll", "text": "evening!"}, {"speaker": "Wren", "text": "Felix, the window table."}]
    assert addressee("ambient_reply", "Doll", {"name": "Wren"}, thread, "Felix") == "Wren"
    assert addressee("ambient_answer", "Doll", {"name": "Wren"}, thread, "Felix") == "Doll"
    assert strip_markers('{"secret":"open"} {"responding_to":"Kit"}Fine, Kit.') == "Fine, Kit."


def test_a_llama_pool_never_masquerades_as_the_overflow_state():
    """2026-10-18: LlamaPool.view() made /health report a list as "overflow", and the capacity test refused to run."""
    from npc_rag.llm import LlamaPool, LlamaClient
    pool = LlamaPool([LlamaClient("http://a"), LlamaClient("http://b")])
    assert not hasattr(pool, "view"), "/health reads view() as the overflow state"
    assert [c["url"] for c in pool.copies()] == ["http://a", "http://b"]
