"""Command line: python -m npc_rag <command>. Run with --help for the list."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .config import ROOT, load_settings


def _embedder(s):
    from .embedder import Embedder
    return Embedder(s.embed_model, s.embed_cache, s.embed_threads)


def cmd_build_index(s, args) -> int:
    from .service import build_index
    idx = build_index(s, _embedder(s), args.source)
    print(json.dumps(idx.meta, indent=1))
    return 0


def cmd_serve(s, args) -> int:
    import uvicorn
    from .service import create_app
    uvicorn.run(create_app(s), host=s.host, port=s.port, log_level="info")
    return 0


def cmd_search(s, args) -> int:
    from .index import WikiIndex
    idx = WikiIndex.load(s.index_dir, _embedder(s))
    hits, best = idx.search(args.query, args.k)
    print(f"best dense {best:.3f} (relevant >= {s.min_relevance})")
    for h in hits:
        print(f"\n[{h.dense:.3f} dense | {h.keyword:.1f} bm25] {h.chunk.page_title} > {h.chunk.heading}\n  {h.chunk.url}\n  {h.chunk.text[:300]}")
    return 0


def cmd_ask(s, args) -> int:
    """Send one message to the running service, like the game server will."""
    import httpx
    headers = {"X-Api-Key": s.api_key} if s.api_key else {}
    body = {"message": args.message, "player_id": args.player, "player_name": args.name}
    r = httpx.post(f"http://127.0.0.1:{s.port}/v1/npc/chat", json=body, headers=headers, timeout=120)
    print(json.dumps(r.json(), indent=1, ensure_ascii=False))
    return 0 if r.status_code == 200 else 1


def cmd_eval(s, args) -> int:
    """Accuracy of the classifier (CLASSIFIER_URL, wiki score as fallback) on data/eval_messages.json."""
    from .classifier import CLASSIFY_PROMPT, classify, parse_verdict
    from .index import WikiIndex
    from .llm import LlamaClient, LLMError

    idx = WikiIndex.load(s.index_dir, _embedder(s))
    llm = LlamaClient(s.classifier_url, s.classifier_model, s.llm_api_key)
    data = json.loads((ROOT / "data" / "eval_messages.json").read_text(encoding="utf-8"))

    async def evidence(q):
        return idx.search(q, 1)[1]

    async def ask(q):
        if args.wiki_only:
            return None
        try:
            return parse_verdict(await llm.chat([{"role": "system", "content": CLASSIFY_PROMPT},
                                                 {"role": "user", "content": q}],
                                                max_tokens=4, temperature=0, timeout=s.classifier_timeout))
        except LLMError as exc:
            print("  classifier failed:", exc)
            return None

    async def run():
        wrong, methods, total = [], {}, 0
        for label in ("game", "general"):
            for msg in data[label]:
                v = await classify(msg, ask_llm=ask, evidence=evidence, wiki_evidence=s.rag_game_evidence)
                total += 1
                methods[v.method] = methods.get(v.method, 0) + 1
                if v.category != label:
                    wrong.append((label, v.method, msg))
        await llm.aclose()
        return wrong, methods, total

    wrong, methods, total = asyncio.run(run())
    print(f"accuracy {total - len(wrong)}/{total} = {(total - len(wrong)) / total:.1%}   decided by {methods}")
    for label, method, msg in wrong:
        print(f"  expected {label:7} ({method}): {msg}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m npc_rag", description="Tutorial NPC RAG for LiDollQuest")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="fetch the wiki and (re)build data/index")
    b.add_argument("--source", help="wiki URL or local web/wiki folder (default: WIKI_SOURCE)")

    sub.add_parser("serve", help="run the service on NPC_PORT (default 9092)")

    se = sub.add_parser("search", help="search the local index from the command line")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=4)

    a = sub.add_parser("ask", help="send a message to the running service")
    a.add_argument("message")
    a.add_argument("--player", default="cli-test", help="player id (keeps a short memory)")
    a.add_argument("--name", default="Tester", help="player name")

    e = sub.add_parser("eval", help="classifier accuracy on data/eval_messages.json")
    e.add_argument("--wiki-only", action="store_true", help="skip the classifier model; wiki score only")

    args = p.parse_args(argv)
    s = load_settings()
    commands = {"build-index": cmd_build_index, "serve": cmd_serve, "search": cmd_search, "ask": cmd_ask, "eval": cmd_eval}
    return commands[args.cmd](s, args)


if __name__ == "__main__":
    sys.exit(main())
