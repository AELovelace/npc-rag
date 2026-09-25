import shutil
import subprocess
from pathlib import Path

import pytest

from npc_rag.wiki import chunk_pages, inline_text, load_pages, page_url, slugify

FIXTURE_WIKI = Path(__file__).parent / "fixtures" / "wiki"

HEADINGS = ["Toilets and other care facilities", "Levels, stats & learning", "Faith and piety",
            "Wearing and changing protection", "Café — Grog's shelf", "What's RPP?"]


def test_slugify_matches_known_ids():
    assert slugify("Toilets and other care facilities") == "toilets-and-other-care-facilities"
    assert slugify("Levels, stats & learning") == "levels-stats--learning"  # wiki.js keeps the double hyphen.


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_slugify_matches_wiki_js():
    """Run the exact regex from web/wiki/wiki.js so heading links can't drift apart."""
    js = ("const s=v=>v.toLowerCase().replace(/[^\\p{L}\\p{N}\\s_-]/gu,'').replace(/ /g,'-');"
          "process.stdout.write(JSON.stringify(JSON.parse(process.argv[1]).map(s)))")
    import json
    out = subprocess.run(["node", "-e", js, json.dumps(HEADINGS)], capture_output=True, text=True, encoding="utf-8", check=True).stdout
    assert json.loads(out) == [slugify(h) for h in HEADINGS]


def test_page_url_matches_route_link():
    assert page_url("https://w/wiki/", "index") == "https://w/wiki/#/"
    assert page_url("https://w/wiki/", "index", "find-a-guide") == "https://w/wiki/#//find-a-guide"
    assert page_url("https://w/wiki", "needs", "levels-stats--learning") == "https://w/wiki/#/needs/levels-stats--learning"


def test_inline_text_strips_markdown():
    assert inline_text("Press **E** near [the toilet](x.md) or `Tab`") == "Press E near the toilet or Tab"


def test_chunks_have_sections_tables_and_duplicate_ids():
    chunks = chunk_pages(load_pages(str(FIXTURE_WIKI)), "https://wiki.test/wiki/", 60)
    ids = [c.id for c in chunks]
    assert "needs#toilets-and-other-care-facilities:0" in ids
    assert "needs#levels-stats--learning:0" in ids and "needs#levels-stats--learning-1:0" in ids  # -1 suffix like wiki.js.
    toilets = next(c for c in chunks if c.id.startswith("needs#toilets"))
    assert "Arcadia: Pay toilets, normally 5 LiDollCoins per use." in toilets.text  # Table row became a sentence.
    assert "Wiki home" not in " ".join(c.text for c in chunks)                      # Navigation row dropped.
    assert not any(c.heading == "not a heading" for c in chunks)                     # Code fences aren't headings.
    overview = next(c for c in chunks if c.id == "index#top:0")
    assert overview.url == "https://wiki.test/wiki/#/" and "Your companion" in overview.text


def test_long_sections_split_with_overlap():
    from npc_rag.wiki import Page
    body = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(6))
    chunks = chunk_pages([Page("p", "P", "", "## Long\n\n" + body)], "https://w/", 70)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 70 + 32 for c in chunks)             # Soft cap plus one carried block.
    assert chunks[1].text.split("\n")[0] == chunks[0].text.split("\n")[-1]  # Overlap keeps context.


def test_missing_wiki_folder_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pages(str(tmp_path))
