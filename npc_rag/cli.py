"""Command line: python -m npc_rag <command>. Run with --help for the list."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

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
    """Evaluate untouched held-out labels, optionally using a live RAG index and extra mixed-intent cases."""
    import httpx
    from .classifier import CLASSIFIER_REVISION, classify
    from .classifier_backend import make_classifier
    from .evaluation import summarize
    from .index import WikiIndex
    from .llm import LLMError

    logging.getLogger("httpx").setLevel(logging.WARNING)  # Accuracy summaries should not be buried in per-request logs.
    idx = None if args.evidence_url or args.raw_model else WikiIndex.load(s.index_dir, _embedder(s))
    if args.classifier_backend:
        s = replace(s, classifier_backend=args.classifier_backend)
    if args.classifier_url:
        if s.classifier_backend == "decider":
            raise ValueError("--classifier-url requires --classifier-backend llamacpp")
        s = replace(s, classifier_url=args.classifier_url)
    llm = None
    if getattr(args, 'decider_torch_model', None):
        if args.wiki_only or s.classifier_backend != 'decider':
            raise ValueError('--decider-torch-model requires Decider model evaluation')
        from .classifier_backend import make_torch_eval_classifier
        llm = make_torch_eval_classifier(s, args.decider_torch_model)
    elif not args.wiki_only:
        llm = make_classifier(s)  # Wiki-only evaluation never downloads or loads Decider.
    data = json.loads((ROOT / "data" / "eval_messages.json").read_text(encoding="utf-8"))
    cases = [(label, msg, "held_out") for label in ("game", "general") for msg in data[label]]
    if args.include_mixed:
        extra = json.loads((ROOT / "data" / "eval_mixed_messages.json").read_text(encoding="utf-8"))
        cases.extend((label, msg, "mixed_intent") for label in ("game", "general") for msg in extra[label])
    remote = httpx.AsyncClient(base_url=args.evidence_url or "http://unused",
                              headers={"X-Api-Key": s.api_key}, timeout=15)

    async def evidence(q):
        if args.evidence_url:
            response = await remote.post("/v1/search", json={"query": q, "k": s.top_k})
            response.raise_for_status()
            found = response.json()  # Use the deployed wiki without changing any remote settings.
            return {"dense": found["best_dense"], "coverage": found.get("keyword_coverage")}  # Older services omit coverage.
        _, best, coverage = await asyncio.to_thread(idx.search_evidence, q, s.top_k)
        return {"dense": best, "coverage": coverage}

    async def ask(q):
        if args.wiki_only:
            return None
        try:
            return await asyncio.wait_for(llm.decide(q), timeout=s.classifier_timeout)
        except (LLMError, asyncio.TimeoutError):
            return None

    async def run():
        rows = []
        try:
            if llm is not None:
                await llm.start()
            identity = await llm.model_identity() if llm is not None else None
            for label, msg, dataset in cases:
                start = time.perf_counter()
                row = {"expected": label, "message": msg, "dataset": dataset}
                try:
                    if args.raw_model:
                        decision = await ask(msg)
                        row.update(category=decision, method="raw_model")
                        if hasattr(decision, 'game_probability'):
                            row['game_probability'] = decision.game_probability
                    else:
                        row.update((await classify(msg, ask_llm=ask, evidence=evidence, wiki_evidence=s.rag_game_evidence,
                                                   keyword_coverage=s.rag_keyword_coverage,
                                                   keyword_dense=s.rag_keyword_dense,
                                                   wiki_first=s.classifier_backend == "decider")).to_dict())  # Same order as the service.
                except Exception as exc:
                    row.update(category=None, method="error", error=type(exc).__name__)  # Failed requests count as wrong.
                row["ms"] = round((time.perf_counter() - start) * 1000)
                rows.append(row)
            return {"policy": CLASSIFIER_REVISION, "model_identity": identity,
                    "classifier_url": s.classifier_url if s.classifier_backend == "llamacpp" else None,
                    "classifier": llm.view() if llm is not None else {"backend": "wiki-only"},
                    "evidence_url": args.evidence_url,
                    "wiki_evidence": s.rag_game_evidence, "keyword_coverage": s.rag_keyword_coverage,
                    "keyword_dense": s.rag_keyword_dense, "raw_model": args.raw_model, "wiki_only": args.wiki_only,
                    "summary": summarize(rows),
                    "datasets": {name: summarize([row for row in rows if row["dataset"] == name])
                                 for name in sorted({row["dataset"] for row in rows})}, "results": rows}
        finally:
            if llm is not None:
                await llm.aclose()
            await remote.aclose()

    report = asyncio.run(run())
    print(json.dumps(report["summary"], indent=2))
    for row in report["results"]:
        if row.get("category") != row["expected"]:
            print(f"  expected {row['expected']:7} ({row.get('category')}/{row['method']}): {row['message']}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Artifact:", output)
    return 0 if all(item["accuracy"] is not None and item["accuracy"] >= args.min_class_accuracy
                    for item in report["summary"]["by_class"].values()) else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m npc_rag", description="Tutorial NPC RAG for LiDollQuest")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="fetch the wiki and (re)build data/index")
    b.add_argument("--source", help="wiki URL or local web/wiki folder (default: WIKI_SOURCE)")

    serve = sub.add_parser("serve", help="run the service on NPC_PORT (default 9092)")
    serve.add_argument("--env-file", help="optional alternate environment file; defaults to .env")  # No secrets enter process arguments.
    sub.add_parser("devices", help="list native GPU names/descriptions without loading classifier weights")

    se = sub.add_parser("search", help="search the local index from the command line")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=4)

    a = sub.add_parser("ask", help="send a message to the running service")
    a.add_argument("message")
    a.add_argument("--player", default="cli-test", help="player id (keeps a short memory)")
    a.add_argument("--name", default="Tester", help="player name")

    e = sub.add_parser("eval", help="classifier accuracy on data/eval_messages.json")
    e.add_argument("--env-file", help="optional environment file, as with serve")
    e.add_argument("--classifier-backend", choices=("llamacpp", "decider"), help="override CLASSIFIER_BACKEND for this evaluation only")
    e.add_argument("--wiki-only", action="store_true", help="skip the classifier model; wiki score only")
    e.add_argument("--raw-model", action="store_true", help="measure model decisions without social shortcuts or wiki overrides")
    e.add_argument("--classifier-url", help="test another model endpoint without changing the service environment")
    e.add_argument('--decider-torch-model', type=Path, help='evaluate local base/merged Torch weights on CUDA (evaluation only)')
    e.add_argument("--evidence-url", help="use /v1/search on this RAG base URL instead of the local wiki index")
    e.add_argument("--include-mixed", action="store_true", help="also check greeting-plus-mechanics and emotional messages")
    e.add_argument("--output", help="write per-message decisions, timing and accuracy to a JSON artifact")
    e.add_argument("--min-class-accuracy", type=float, default=0.9, help="required accuracy for EACH class (default: 0.9)")

    gm = sub.add_parser("gm-build-index", help="build the separate GM handbook index")
    gm.add_argument("--source", help="GM wiki folder or URL (default: GM_WIKI_SOURCE)")
    sub.add_parser("gm-serve", help="run the staff handbook service on GM_PORT (9093)")

    args = p.parse_args(argv)
    if args.cmd == "devices":
        from .decider_devices import detect_devices
        print(json.dumps({"devices": detect_devices()}, indent=2))
        return 0  # Diagnostic works independently of .env and does not start the RAG service.
    if args.cmd == "eval" and (args.raw_model and args.wiki_only or not 0 <= args.min_class_accuracy <= 1):
        p.error("eval needs a threshold from 0 to 1; --raw-model and --wiki-only cannot be combined")
    if args.cmd.startswith("gm-"):
        from .gm import load_gm_settings, build_gm_index, create_gm_app
        settings = load_gm_settings()
        if args.cmd == "gm-build-index":
            index = build_gm_index(settings, _embedder(settings), args.source)
            print(json.dumps({k: v for k, v in index.meta.items() if k != "illustrations"}, indent=1))
        else:
            import uvicorn
            uvicorn.run(create_gm_app(settings), host=settings.host, port=settings.port, log_level="info")
        return 0
    s = load_settings(getattr(args, "env_file", None))
    commands = {"build-index": cmd_build_index, "serve": cmd_serve, "search": cmd_search, "ask": cmd_ask, "eval": cmd_eval}
    return commands[args.cmd](s, args)


if __name__ == "__main__":
    sys.exit(main())
