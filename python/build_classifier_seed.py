"""Build a reviewable synthetic routing seed; never train a model or modify evaluation data."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

# Each pair shares a topic but changes intent. These are authored examples, not player logs.
# Keep a whole family together in any future train/development split, including its paraphrases.
SEEDS = {
    "getting-started": [
        ("The link code expired before I used it. How do I request another?",
         "I'm glad I finally signed in; it's nice to meet you."),
        ("Can I finish making a character if the connection dropped after Begin?",
         "I spent ages picking my character's name because I wanted it to sound pretty."),
        ("Hi Pip! Can I change my starting hub choice before pressing Begin?",
         "Hi Pip! Which town would you personally enjoy living in?"),
        ("Do the stat points from Reroll replace the ones I already assigned during creation?",
         "Picking a class makes me nervous. Could you say something encouraging?"),
        ("wheres the button to replay that welcome card i skipped",
         "Your welcome made this place feel less lonely. Thank you."),
    ],
    "controls": [
        ("How can I turn my character to face left without walking onto the next tile?",
         "You look very serious facing that way. Are you posing for a portrait?"),
        ("Can I change the map zoom while I am playing on a touchscreen?",
         "Everything looks so tiny from up here; it makes me laugh."),
        ("Hello! Which button releases the chat box so movement keys work again?",
         "Hello! I like chatting with you while we wander."),
        ("The minimap is covered in fog. What does that indicate?",
         "That fog gives the map a mysterious look. Tell me a spooky little story."),
        ("how do i move keyboard focus between the two bank columns",
         "I keep tapping my keyboard to the music. What's your favorite tune?"),
    ],
    "combat": [
        ("If my action gauge stays full, can I save up a second attack?",
         "That last attack looked spectacular! I wish you could have seen my face."),
        ("Will opening the item menu stop an enemy's action gauge?",
         "Waiting for that enemy made me jumpy. Could you distract me with a joke?"),
        ("I'm nervous about this battle. Can a support spell target another party member?",
         "I'm nervous about this battle. Please remind me we're in this together."),
        ("What changes when I switch from the front row to the back row?",
         "I like standing behind you. Your cape is fun to watch."),
        ("does a mage gun spend arrows or mp when it fires",
         "If you could invent a ridiculous magic weapon, what would it look like?"),
    ],
    "needs-and-care": [
        ("Does drinking from a well increase the WET meter later?",
         "That well looks peaceful. Would you sit beside it with me?"),
        ("Where on the item display can I check how much absorption remains?",
         "I feel self-conscious about needing protection. Could you reassure me?"),
        ("I'm embarrassed. Does changing remove every Dignity penalty or just some of them?",
         "I'm embarrassed. Please stay with me until I feel ready to keep going."),
        ("In wet-only mode, is the Tum relief button disabled?",
         "I appreciate that you don't judge me when I need a break."),
        ("my stamina recovered but stand still fails; can equipment force crawling",
         "Crawling around with you made this outing surprisingly silly."),
    ],
    "items-and-economy": [
        ("If I put a sword in the bank, can another character on my account withdraw it?",
         "That sword reminds me of a story my friend used to tell."),
        ("Are ingredients counted against the ordinary bag slots?",
         "I like arranging my ingredients by color even when nobody sees them."),
        ("Hi there, what does Not sellable on an item mean?",
         "Hi there, could you invent a funny sales pitch for this old hat?"),
        ("Does the Cursebreaker remove an item's curse or only let me take it off?",
         "If your boots could talk, would they complain about being cursed?"),
        ("why does this dress occupy two equipment slots",
         "This dress makes me feel cheerful. Do you like its color?"),
    ],
    "player-shops": [
        ("Does reopening my closed shop require paying the license fee again?",
         "Opening a little shop has always sounded cozy to me."),
        ("Can customers buy my listed goods while I am signed out?",
         "I wonder what my shopkeeper daydreams about when nobody visits."),
        ("I'm confused. Do goods purchased by a buy order get listed for sale automatically?",
         "I'm confused about what to name my shop. Suggest something whimsical."),
        ("Do I need to unlist a stack before I can take it back into my bag?",
         "My shelves look lovely today. I just wanted to share that with someone."),
        ("can my alt buy from my own shopkeeper",
         "Would you rather run a tiny bakery or a shop full of magical hats?"),
    ],
    "gathering": [
        ("Does another player's harvest prevent me from gathering at the same node?",
         "Picking berries together sounds like a pleasant way to spend the afternoon."),
        ("Do I need to equip a gathering tool before interacting with a resource marker?",
         "I wish I had a tiny basket just for the prettiest leaves."),
        ("Hello friend, when do personal resource nodes become harvestable again?",
         "Hello friend, which woodland smell do you like best?"),
        ("Which region contains sea-glass salt gathering nodes?",
         "Sea-glass salt is such a pretty name. It sounds like a poem."),
        ("why does this resource marker say harvested after i relog",
         "I found a flower and thought of you. That's all I wanted to say."),
    ],
    "crafting": [
        ("Which crafting station should I use for Tailoring recipes?",
         "I like watching you sew. It looks so calming."),
        ("Where in the menu can I open my crafting discovery journal?",
         "My crafting journal would look lovely with little drawings in the margins."),
        ("I'm disappointed with my cooking. Can burnt scraps be eaten?",
         "I'm disappointed with my cooking. Can you cheer me up?"),
        ("Does eating a second bonus meal replace the first meal's bonus?",
         "If I cooked for you, would you prefer something sweet or something savory?"),
        ("where do i interact to use a campfire for cooking",
         "A crackling campfire and a friend nearby are enough to make me happy."),
    ],
    "world-and-dungeons": [
        ("Which regions connect Honeydew Village and LittleBigCity?",
         "I love the name LittleBigCity. It sounds like the start of a fairy tale."),
        ("Is a weekly Dive the same place as the full dungeon with that name?",
         "That dungeon entrance looks intimidating. Would you tell me a brave story?"),
        ("Hey, if I leave through Escape, which entrance will it return me to?",
         "Hey, I missed having you beside me on that long journey."),
        ("Can a merchant's map position change when the town layout refreshes?",
         "The town looks different today. I like noticing little changes with you."),
        ("which way from dustbreak desert leads into echo gulch",
         "Echo Gulch sounds like a good place to shout something silly."),
    ],
    "quests-and-faith": [
        ("Can I track two accepted quests on the HUD at the same time?",
         "Finishing that quest made me proud. Will you celebrate with me?"),
        ("What is the difference between the blue and gold quest markers?",
         "Blue and gold are such cheerful colors together, don't you think?"),
        ("Thanks for helping! Do delivery quests finish just from carrying their item?",
         "Thanks for helping! Your company matters more to me than the reward."),
        ("Does reconnecting let me claim a quest reward that I already collected?",
         "I keep remembering the story from that quest. It made me smile."),
        ("is my first vow free if i skipped choosing a patron during creation",
         "If you could offer a patron one heartfelt wish, what would you ask for?"),
    ],
    "multiplayer": [
        ("Does a hired companion take one of the party's three slots?",
         "I'm happy you're my traveling companion. What do you enjoy about our trips?"),
        ("Can a passenger use the leader's beacon destination without discovering it first?",
         "That glowing beacon makes me think of home. Does it remind you of anything?"),
        ("I'm worried about running out of time. Does my companion rental keep ticking while offline?",
         "I'm worried about saying goodbye. Can we talk for a little while?"),
        ("Which chat channel should I use to speak to a nearby NPC companion?",
         "Tell me about a friend you miss, if you feel like sharing."),
        ("how do i extend the companion hire for another hour",
         "I wish this walk could last another hour. I'm enjoying your stories."),
    ],
    "saving-and-help": [
        ("How can I resume an existing online character from the title screen?",
         "Seeing my character again after a break made me oddly happy."),
        ("Where should I look for the reason an interaction was refused?",
         "Getting stuck can be frustrating. Could you tell me something comforting?"),
        ("Hi Pip, why does saving my Wardrobe look cost diamonds when I am away from a mirror?",
         "Hi Pip, it's nice to have someone patient to talk to."),
        ("Does renaming a character spend coins or stars?",
         "If we both had silly nicknames, what would you call us?"),
        ("what details should i include when reporting a stuck doorway",
         "I think I'll take a break now. Thanks for keeping me company."),
    ],
}


def normalize(message):
    words = " ".join(re.findall(r"\w+", message.casefold()))
    return words or message.strip()  # Preserve distinct emoji-only reactions while ignoring punctuation around words.


def build(wiki_root):
    excluded = set()
    evaluation_hashes = {}
    for name in ("eval_messages.json", "eval_mixed_messages.json"):
        path = ROOT / "data" / name
        raw = path.read_bytes()
        evaluation_hashes[name] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        excluded.update(normalize(message) for label in ("game", "general") for message in data[label])
    rows, seen, sources = [], set(), {}
    for topic, pairs in SEEDS.items():
        path = wiki_root / "content" / (topic + ".md")
        sources[topic] = hashlib.sha256(path.read_bytes()).hexdigest()  # Pin the wiki revision used to check topic coverage.
        for number, pair in enumerate(pairs, 1):
            family = f"{topic}-{number:02d}"
            for label, message in zip(("game", "general"), pair):
                normalized = normalize(message)
                if not message.strip() or len(message) > 500 or normalized in seen or normalized in excluded:
                    raise ValueError(f"Invalid, duplicate or evaluation-overlapping example: {family}/{label}")
                seen.add(normalized)
                rows.append({"id": f"seed-v1-{family}-{label}", "message": message, "label": label,
                             "family_id": family, "topic": topic,
                             "source_page": f"content/{topic}.md", "origin": "assistant_authored_synthetic",
                             "review_status": "needs_review", "split": "unassigned"})
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "purpose": "Draft examples for gameplay-versus-social routing; no answers or model training.",
                "counts": dict(Counter(row["label"] for row in rows)), "total": len(rows),
                "families": len({row["family_id"] for row in rows}), "topics": len(sources),
                "wiki_sha256": sources, "evaluation_sha256": evaluation_hashes,
                "evaluation_overlap_normalized": 0, "player_logs_used": False,
                "independent_label_review": False, "training_ready": False,
                "notes": ["Metadata and labels must never become part of classifier input.",
                          "Keep families and all derived paraphrases in the same future split.",
                          "Topic provenance does not assert that every premise is true or answerable.",
                          "Normalized exclusion is not a semantic leakage check; review near-duplicates separately.",
                          "This starter covers game-related topics; add diverse ordinary chat before training."]}
    return rows, manifest  # Produce explicit draft records without reading private conversations or changing the gate.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-root", type=Path, required=True, help="local web/wiki directory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "classifier")
    args = parser.parse_args()
    rows, manifest = build(args.wiki_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "synthetic-seed-v1.jsonl"
    data_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest["dataset_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    (args.output_dir / "synthetic-seed-v1.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(data_path), "total": len(rows), "counts": manifest["counts"],
                      "training_ready": False}, indent=2))


if __name__ == "__main__":
    main()
