"""Compare Decider readouts on synthetic development cases, without changing the RAG service."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Run from the checkout without installing npc-rag as a package.

from npc_rag.classifier_backend import DECIDER_QUESTION, DeciderClassifier
from npc_rag.config import load_settings

# These development examples are separate from the held-out evaluation datasets.
CASES = [
    ("GAME", "Which button opens the equipment panel in this video game?"),
    ("GAME", "Explain the rules for casting a spell during a combat turn."),
    ("GAME", "Hi friend! How can I assign an ability to a keyboard shortcut?"),
    ("CHAT", "What song would you sing on a cheerful afternoon?"),
    ("CHAT", "I am glad we get to spend a little time together."),
    ("CHAT", "Playing this game with you makes me happy."),
]
SHORT_QUESTION = "Is the player asking for gameplay help, or having a social conversation?"
DESCRIBED = ["Gameplay help: rules, controls, equipment, quests or mechanics",
             "Social conversation: feelings, jokes, preferences or character chat"]
CONTROLS = [
    ("The parcel arrived damaged and I want my money back.", "Which department should handle this?",
     ["refunds", "technical support", "sales"], "refunds"),
    ("I cannot sign in because the application crashes.", "Which department should handle this?",
     ["refunds", "technical support", "sales"], "technical support"),
    ("I want a quote for buying twenty licenses.", "Which department should handle this?",
     ["refunds", "technical support", "sales"], "sales"),
    ("The object is a ripe banana.", "Is this a fruit?", ["yes", "no"], "yes"),
    ("The object is a metal spanner.", "Is this a fruit?", ["yes", "no"], "no"),
]


def package_versions():
    result = {}
    for name in ("decider-ai", "llama-cpp-python", "torch", "transformers", "numpy"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result  # Report runtime identity, never environment variables or credentials.


def investigate(model, context):
    rows = []

    def probe(group, state, question, options, expected, mapping=None):
        start = time.perf_counter()
        row = {"group": group, "state": state, "question": question,
               "options": options, "expected": expected}
        try:
            result = model.decide(state, [{"question": question, "options": options}],
                                  max_ctx_tokens=context - 1024)[0]
            row["result"] = result  # Preserve both probabilities and the upstream winning choice.
            row["predicted"] = (mapping or {}).get(result["choice"], result["choice"])
            row["correct"] = row["predicted"] == expected
        except Exception as exc:
            row.update(error=type(exc).__name__, correct=False)
        row["ms"] = round((time.perf_counter() - start) * 1000)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    for state, question, options, expected in CONTROLS:
        probe("control", state, question, options, expected)
        probe("control_reversed", state, question, list(reversed(options)), expected)
    variants = [
        ("production", DECIDER_QUESTION, ["GAME", "CHAT"], True, None),
        ("production_reversed", DECIDER_QUESTION, ["CHAT", "GAME"], True, None),
        ("short_described_json", SHORT_QUESTION, DESCRIBED, True, dict(zip(DESCRIBED, ["GAME", "CHAT"]))),
        ("short_described_text", SHORT_QUESTION, DESCRIBED, False, dict(zip(DESCRIBED, ["GAME", "CHAT"]))),
    ]
    for group, question, options, quoted, mapping in variants:
        for expected, message in CASES:
            state = json.dumps({"message": message}, ensure_ascii=False) if quoted else message
            probe(group, state, question, options, expected, mapping)
    # Repeat the first control after unrelated questions to detect state-dependent drift.
    state, question, options, expected = CONTROLS[0]
    probe("repeat_control", state, question, options, expected)
    return rows


async def run(args):
    settings = load_settings(args.env_file)
    backend = DeciderClassifier(settings)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "platform": platform.platform(), "packages": package_versions(), "backend": backend.view(),
              "held_out_cases_used": False}
    try:
        await backend.start()
        model = backend._model  # Diagnostic-only inspection of the same loader used by production.
        import llama_cpp
        report["native_system_info"] = llama_cpp.llama_print_system_info().decode("utf-8", errors="replace")
        report["readout"] = {"layout": model.layout, "temperature": model.T,
                             "temperature_by_type": model.T_by_type,
                             "option_letter_ids": model.eng.letters[:10].tolist(),
                             "tokenizer_A": model.m.tok.encode("A", add_special_tokens=False),
                             "tokenizer_B": model.m.tok.encode("B", add_special_tokens=False)}
        # The same worker owns all probes. This separate process has no access to the live cache/context.
        report["results"] = await backend._native(lambda: investigate(model, settings.decider_context))
        groups = sorted({row["group"] for row in report["results"]})
        report["summary"] = {}
        for group in groups:
            rows = [row for row in report["results"] if row["group"] == group]
            report["summary"][group] = {"correct": sum(row["correct"] for row in rows), "total": len(rows),
                                        "predicted": dict(Counter(row.get("predicted", "error") for row in rows))}
    finally:
        await backend.aclose()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Artifact:", output.resolve(), flush=True)  # Synthetic inputs only; no API keys or player logs.
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0 if all(row["correct"] for row in report["results"] if row["group"].startswith("control")) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", help="use the same RAG environment file as the running service")
    parser.add_argument("--output", default="artifacts/decider-diagnostic.json")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
