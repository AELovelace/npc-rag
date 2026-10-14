"""Build 1,000 synthetic rows with family-safe splits and recorded augmentation provenance."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random

from build_classifier_seed import ROOT, SEEDS, normalize
from classifier_expansion import EXTRA

PREFIXES = ("hey friend, ", "quick thing: ", "so, ", "one more thing: ", "please hear me out: ")


def partition(rows, seed=417):
    families = sorted({r["family_id"] for r in rows})
    parent = {f: f for f in families}

    def root(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    authored = [r for r in rows if r["variant"] == "authored"]
    matches = []
    for i, a in enumerate(authored):
        for b in authored[i + 1:]:
            if a["family_id"] == b["family_id"]:
                continue
            similarity = SequenceMatcher(None, normalize(a["message"]), normalize(b["message"])).ratio()
            if similarity >= .82:
                parent[root(b["family_id"])] = root(a["family_id"])
                matches.append({"a": a["id"], "b": b["id"], "similarity": round(similarity, 4)})
    groups = defaultdict(list)
    for row in rows:
        groups[root(row["family_id"])].append(row)
    remaining = list(sorted(groups))
    random.Random(seed).shuffle(remaining)

    def select(target):
        paths = {0: []}
        for key in remaining:
            for count, picked in list(paths.items()):
                total = count + len(groups[key])
                if total <= target and total not in paths:
                    paths[total] = picked + [key]
        if target not in paths:
            raise ValueError("Related families cannot fit requested split sizes; adjust the dataset, not the grouping")
        return set(paths[target])  # Choose whole similarity-connected groups, never individual rows.

    for split, count in (("test", 100), ("validation", 100)):
        chosen = select(count)
        for key in chosen:
            for row in groups[key]:
                row.update(split=split, split_group=key)
        remaining = [key for key in remaining if key not in chosen]
    for key in remaining:
        for row in groups[key]:
            row.update(split="train", split_group=key)
    return matches  # Similarity grouping reduces textual leakage, but cannot prove semantic independence.


def build(wiki_root):
    heldout, source_hashes, evaluation_hashes = set(), {}, {}
    for name in ("eval_messages.json", "eval_mixed_messages.json"):
        path = ROOT / "data" / name
        evaluation_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text(encoding="utf-8"))
        heldout.update(normalize(s) for label in ("game", "general") for s in data[label])
    rows, seen = [], set()
    for topic, base in SEEDS.items():
        source_hashes[topic] = hashlib.sha256((wiki_root / "content" / (topic + ".md")).read_bytes()).hexdigest()
        for number, pair in enumerate(base + EXTRA[topic], 1):
            family = f"{topic}-{number:02d}"
            for label, message in zip(("game", "general"), pair):
                variant = PREFIXES[(number - 1) % len(PREFIXES)] + message[0].lower() + message[1:]
                for kind, text in (("authored", message), ("conversational_prefix", variant)):
                    normalized = normalize(text)
                    if not text.strip() or len(text) > 500 or normalized in seen or normalized in heldout:
                        raise ValueError(f"Duplicate, evaluation overlap or invalid message: {family}/{label}/{kind}")
                    seen.add(normalized)
                    rows.append({"id": f"pilot-v1-{family}-{label}-{kind}", "message": text, "label": label,
                                 "family_id": family, "topic": topic, "variant": kind,
                                 "source_page": f"content/{topic}.md", "origin": "assistant_authored_synthetic",
                                 "review_status": "synthetic_draft"})
    if len(rows) != 1000:
        raise ValueError(f"Expected 1,000 rows; found {len(rows)}")
    matches = partition(rows)
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "total": len(rows), "authored_messages": 500, "prefix_augmented_messages": 500,
                "families": 250, "split_seed": 417, "labels": dict(Counter(r["label"] for r in rows)),
                "splits": {s: dict(Counter(r["label"] for r in rows if r["split"] == s))
                           for s in ("train", "validation", "test")},
                "wiki_sha256": source_hashes, "evaluation_sha256": evaluation_hashes,
                "near_duplicate_grouping_threshold": .82, "grouped_similar_pairs": matches,
                "evaluation_overlap_normalized": 0, "player_logs_used": False,
                "independent_label_review": False, "release_qualified": False,
                "notes": ["1,000 rows represent 250 paired families, not 1,000 independent intents.",
                          "500 authored messages plus 500 explicit prefix augmentations; no hidden bulk paraphrasing.",
                          "Train/validation/test are 800/100/100 with whole families and similar pairs kept together.",
                          "Synthetic validation guides the experiment; final qualification needs reviewed real traffic.",
                          "Only message text enters the model. Labels, topic, source and split never enter prompts.",
                          "No unrelated ordinary-chat corpus or generic-task replay is included in this first pilot."]}
    return rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "classifier")
    args = parser.parse_args()
    rows, manifest = build(args.wiki_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "synthetic-pilot-v1.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    manifest["dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output_dir / "synthetic-pilot-v1.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(path), "total": len(rows), "splits": manifest["splits"],
                      "similar_pairs_grouped": len(manifest["grouped_similar_pairs"])}, indent=2))


if __name__ == "__main__":
    main()
