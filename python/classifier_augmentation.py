"""Label-independent surface augmentation; sample once per example per epoch."""
from collections import Counter
import random
import re

PREFIXES = ("hey, ", "so ", "um ", "quick thing ", "please ")
REPLACEMENTS = {"how": "hw", "level": "lvl", "you": "u", "please": "pls",
                "right now": "rn", "the": "teh", "what": "wat", "where": "wher"}


def augment_rows(rows, seed):
    rng = random.Random(seed)
    output = []
    for row in rows:
        text, variants = row["message"], []
        if rng.random() < .9:
            text = text.lower(); variants.append("lowercase")
        if rng.random() < .85:
            text = text.rstrip("?.! "); variants.append("drop_trailing_punctuation")
        if rng.random() < .5:
            text = text.replace("'", "").replace("’", ""); variants.append("strip_apostrophes")
        if rng.random() < .45:
            for word, replacement in REPLACEMENTS.items():
                if rng.random() < .5 and re.search(r"\b" + word + r"\b", text):
                    text = re.sub(r"\b" + word + r"\b", replacement, text)
                    variants.append("text_" + word.replace(" ", "_"))
        if rng.random() < .12:
            text = rng.choice(PREFIXES) + text; variants.append("conversational_prefix")
        if rng.random() < .3:
            text = text.rstrip("?.! ") + "?"; variants.append("add_question_mark")
        output.append({**row, "message": text, "variant": variants or ["unchanged"]})
    return output  # IDs/families remain stable; augmentation never adds rows or changes a label.


def shortcut_stats(rows):
    correct = sum(("game" if "?" in r["message"] and "you" not in r["message"].lower() else "general")
                  == r["label"] for r in rows)
    return {"rule": "contains ? and no you substring => game; otherwise general",
            "correct": correct, "total": len(rows), "accuracy": correct / len(rows)}


def check_shortcut(rows, ceiling=.65):
    stats = shortcut_stats(rows)
    if stats["accuracy"] > ceiling:
        raise ValueError(f"Punctuation shortcut exceeds {ceiling:.0%}: {stats['accuracy']:.1%}")
    return stats


def augmentation_stats(rows):
    return {"shortcut": check_shortcut(rows),
            "operations_by_label": {label: dict(Counter(v for r in rows if r["label"] == label for v in r["variant"]))
                                    for label in ("game", "general")}}
