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
    from .rag_service import build_index
    idx = build_index(s, _embedder(s), args.source)
    print(json.dumps(idx.meta, indent=1))
    return 0


def cmd_serve(s, args) -> int:
    import uvicorn
    if args.service == "rag":
        from .rag_service import create_app
        uvicorn.run(create_app(s), host=s.rag_host, port=s.rag_port, log_level="info")
    else:
        from .agent_service import create_app
        uvicorn.run(create_app(s), host=s.agent_host, port=s.agent_port, log_level="info")
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
    """Send one message to the running agent service, like the game server will."""
    import httpx
    headers = {"X-Api-Key": s.api_key} if s.api_key else {}
    body = {"message": args.message, "player_id": args.player, "player_name": args.name}
    r = httpx.post(f"http://127.0.0.1:{s.agent_port}/v1/npc/chat", json=body, headers=headers, timeout=120)
    print(json.dumps(r.json(), indent=1, ensure_ascii=False))
    return 0 if r.status_code == 200 else 1


def cmd_eval(s, args) -> int:
    """Accuracy of the classifier on data/eval_messages.json (held out from the seeds)."""
    from .classifier import TIEBREAK_PROMPT, Classifier, load_seeds, parse_tiebreak
    from .index import WikiIndex
    from .llm import LlamaClient, LLMError

    emb = _embedder(s)
    idx = WikiIndex.load(s.index_dir, emb)
    llm = LlamaClient(s.llm_url, s.llm_model, s.llm_api_key) if args.llm else None

    async def evidence(q):
        return idx.search(q, 1)[1]

    async def tiebreak(q):
        try:
            return parse_tiebreak(await llm.chat([{"role": "system", "content": TIEBREAK_PROMPT},
                                                  {"role": "user", "content": q}], max_tokens=4, temperature=0, timeout=20))
        except LLMError as exc:
            print("  tiebreak failed:", exc)
            return None

    game, general = load_seeds(s.seeds_path)
    clf = Classifier(emb, game, general, margin=s.classifier_margin, wiki_evidence=s.rag_game_evidence,
                     evidence=evidence, tiebreak=tiebreak if llm else None)
    data = json.loads((ROOT / "data" / "eval_messages.json").read_text(encoding="utf-8"))

    async def run():
        wrong, methods, total = [], {}, 0
        for label in ("game", "general"):
            for msg in data[label]:
                v = await clf.classify(msg)
                total += 1
                methods[v.method] = methods.get(v.method, 0) + 1
                if v.category != label:
                    wrong.append((label, v.method, msg))
        if llm:
            await llm.aclose()
        return wrong, methods, total

    wrong, methods, total = asyncio.run(run())
    print(f"accuracy {total - len(wrong)}/{total} = {(total - len(wrong)) / total:.1%}   decided by {methods}")
    for label, method, msg in wrong:
        print(f"  expected {label:7} ({method}): {msg}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m npc_rag", description="Tutorial NPC RAG agent for LiDollQuest")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="fetch the wiki and (re)build data/index")
    b.add_argument("--source", help="wiki URL or local web/wiki folder (default: WIKI_SOURCE)")

    sv = sub.add_parser("serve", help="run a service")
    sv.add_argument("service", choices=["agent", "rag"], help="agent = classifier + chat on 9091, rag = retrieval on 9092")

    se = sub.add_parser("search", help="search the local index from the command line")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=4)

    a = sub.add_parser("ask", help="send a message to the running agent service")
    a.add_argument("message")
    a.add_argument("--player", default="cli-test", help="player id (keeps a short memory)")
    a.add_argument("--name", default="Tester", help="player name")

    e = sub.add_parser("eval", help="classifier accuracy on data/eval_messages.json")
    e.add_argument("--llm", action="store_true", help="include the llama.cpp tiebreak step")

    args = p.parse_args(argv)
    s = load_settings()
    commands = {"build-index": cmd_build_index, "serve": cmd_serve, "search": cmd_search, "ask": cmd_ask, "eval": cmd_eval}
    return commands[args.cmd](s, args)


if __name__ == "__main__":
    sys.exit(main())
