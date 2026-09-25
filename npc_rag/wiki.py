"""Load the public LiDollQuest wiki (URL or local web/wiki folder) and cut it into retrievable chunks."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import httpx


@dataclass
class Page:
    slug: str         # pages.json slug, also the content/<slug>.md file name.
    title: str        # Chapter title shown in the wiki navigation.
    description: str  # One-line summary from pages.json.
    markdown: str     # Raw Markdown text of the chapter.


@dataclass
class Chunk:
    id: str           # Stable id: "<slug>#<anchor>:<part>".
    slug: str         # Chapter slug.
    page_title: str   # Chapter title.
    heading: str      # Section heading path, e.g. "Toilets and other care facilities".
    url: str          # Deep link into the public wiki.
    text: str         # Plain-text body used for both BM25 and embeddings.

    def embed_text(self) -> str:
        """Text given to the embedding model: chapter + heading give short sections their context."""
        return f"{self.page_title} - {self.heading}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Loading ──────────────────────────────────────────────────────────────────

def load_pages(source: str, timeout: float = 20.0) -> list[Page]:
    """Read pages.json and every chapter from a wiki URL or a local web/wiki folder."""
    if re.match(r"^https?://", source, re.I):
        base = source if source.endswith("/") else source + "/"  # pages.json and content/ sit under the wiki root.
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            listing = client.get(base + "pages.json")
            listing.raise_for_status()  # A missing wiki is a hard error: never build an empty index silently.
            pages = []
            for entry in listing.json():
                r = client.get(f"{base}content/{entry['slug']}.md")  # Same path the browser wiki fetches.
                r.raise_for_status()
                pages.append(_page(entry, r.text))
            return pages

    root = Path(source)
    if (root / "web" / "wiki" / "pages.json").is_file():
        root = root / "web" / "wiki"  # Allow pointing at the lidollquest repo root as well.
    listing_path = root / "pages.json"
    if not listing_path.is_file():
        raise FileNotFoundError(f"No pages.json in {root}; point WIKI_SOURCE at web/wiki or the wiki URL")
    entries = json.loads(listing_path.read_text(encoding="utf-8"))
    return [_page(e, (root / "content" / f"{e['slug']}.md").read_text(encoding="utf-8")) for e in entries]


def _page(entry: dict, markdown: str) -> Page:
    return Page(entry["slug"], entry.get("title", entry["slug"]), entry.get("description", ""), markdown)


# ── Markdown to plain text ───────────────────────────────────────────────────

_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")        # [text](url) and ![alt](img)
_EMPH = re.compile(r"(\*\*|__|\*|_|`)(?=\S)(.+?)(?<=\S)\1")  # **bold**, *em*, `code`
_TAG = re.compile(r"<[^>]+>")                          # Stray inline HTML.


def inline_text(md: str) -> str:
    """Strip inline Markdown the way a rendered heading/paragraph's textContent would read."""
    text = _LINK.sub(r"\1", md)            # Keep link text, drop the target.
    for _ in range(2):
        text = _EMPH.sub(r"\2", text)      # Twice handles nesting like ***bold italic***.
    text = _TAG.sub("", text)
    return text.replace("\\|", "|").strip()


def slugify(heading_text: str) -> str:
    """Python twin of wiki.js slugify(): lowercase, drop punctuation, spaces become hyphens."""
    lowered = heading_text.lower()
    kept = re.sub(r"[^\w\s-]", "", lowered)  # \w is letters, digits and _ (Unicode aware), like \p{L}\p{N}_.
    return kept.replace(" ", "-")            # wiki.js only replaces literal spaces.


def page_url(public_base: str, slug: str, anchor: str = "") -> str:
    """Deep link matching wiki.js routeLink(): #/<slug>/<encoded anchor>, with the home page as #/."""
    base = public_base if public_base.endswith("/") else public_base + "/"
    route = "" if slug == "index" else slug
    tail = "/" + quote(anchor, safe="-_.!~*'()") if anchor else ""  # Same safe set as encodeURIComponent.
    return f"{base}#/{route}{tail}"


def _table_rows(lines: list[str]) -> list[str]:
    """Turn a Markdown table into one readable sentence per row."""
    cells = [[inline_text(c) for c in line.strip().strip("|").split("|")] for line in lines]
    cells = [row for row in cells if not all(re.fullmatch(r":?-{2,}:?", c.strip()) or not c for c in row)]  # Drop the --- row.
    if not cells:
        return []
    header, rows = cells[0], cells[1:]
    out = []
    for row in rows:
        if len(row) == 2:
            out.append(f"{row[0]}: {row[1]}")  # Two columns read as "term: meaning".
        else:
            rest = "; ".join(f"{h}: {v}" for h, v in zip(header[1:], row[1:]) if v)
            out.append(f"{row[0]} ({rest})" if rest else row[0])
    return out


def _blocks(body_lines: list[str]) -> list[str]:
    """Group section lines into paragraphs, list items and table rows (the units a chunk may split on)."""
    blocks: list[str] = []
    para: list[str] = []
    table: list[str] = []

    def flush_para():
        if para:
            blocks.append(inline_text(" ".join(para)))
            para.clear()

    def flush_table():
        if table:
            blocks.extend(_table_rows(table))
            table.clear()

    for line in body_lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            flush_para()
            table.append(stripped)  # Collect the whole table before converting it.
            continue
        flush_table()
        if not stripped or stripped.startswith("```"):
            flush_para()  # Blank lines and code fences end a paragraph.
        elif re.match(r"^([-*+]|\d+[.)])\s+", stripped):
            flush_para()
            blocks.append(inline_text(re.sub(r"^([-*+]|\d+[.)])\s+", "", stripped)))  # Each list item is its own block.
        elif stripped.startswith(">"):
            para.append(stripped.lstrip("> "))  # Blockquotes read as normal text.
        else:
            para.append(stripped)
    flush_para()
    flush_table()
    return [b for b in blocks if b]


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_pages(pages: list[Page], public_base: str, max_words: int = 180) -> list[Chunk]:
    """Split every chapter at its headings; long sections become several chunks of about max_words."""
    chunks: list[Chunk] = []
    for page in pages:
        lines = page.markdown.splitlines()
        # wiki.js prepareMarkdown() drops the "# Title" line and the "[Wiki home] ..." navigation row.
        while lines and (not lines[0].strip() or lines[0].startswith("# ") or lines[0].startswith("[Wiki home]")):
            lines.pop(0)

        seen: dict[str, int] = {}                 # Duplicate heading ids get -1, -2 ... like wiki.js.
        sections: list[tuple[str, str, list[str]]] = [("Overview", "", [])]  # Text before the first heading.
        in_code = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code = not in_code             # Never treat "## " inside a code block as a heading.
            m = None if in_code else re.match(r"^(#{2,6})\s+(.*?)\s*#*\s*$", line)
            if m:
                title = inline_text(m.group(2))
                base = slugify(title)
                count = seen.get(base, 0)
                seen[base] = count + 1
                anchor = base + (f"-{count}" if count else "")
                sections.append((title, anchor, []))
            else:
                sections[-1][2].append(line)

        for heading, anchor, body in sections:
            blocks = _blocks(body)
            if heading == "Overview" and page.description:
                blocks.insert(0, page.description)  # The chapter summary helps broad questions find the page.
            if not blocks:
                continue                           # Headings that only introduce subsections carry no text.
            url = page_url(public_base, page.slug, anchor)
            for part, text in enumerate(_pack(blocks, max_words)):
                chunks.append(Chunk(f"{page.slug}#{anchor or 'top'}:{part}", page.slug, page.title, heading, url, text))
    return chunks


def _pack(blocks: list[str], max_words: int) -> list[str]:
    """Greedily pack blocks into chunks; the last block of a full chunk is repeated for overlap."""
    out: list[str] = []
    current: list[str] = []
    words = 0
    for block in blocks:
        n = len(block.split())
        if current and words + n > max_words:
            out.append("\n".join(current))
            current = [current[-1]] if len(current[-1].split()) < max_words // 2 else []  # Carry one block of context.
            words = sum(len(b.split()) for b in current)
        current.append(block)
        words += n
    if current:
        out.append("\n".join(current))
    return out
