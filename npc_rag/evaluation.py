"""Summaries for classifier accuracy checks; keep failures and the two error directions visible."""
from collections import Counter

from .classifier import GAME, GENERAL


def summarize(rows: list[dict]) -> dict:
    """Report per-class recall so a classifier cannot pass by always choosing the majority class."""
    by_class = {}
    for label in (GAME, GENERAL):
        subset = [row for row in rows if row["expected"] == label]
        correct = sum(row.get("category") == label for row in subset)
        by_class[label] = {"total": len(subset), "correct": correct,
                           "accuracy": correct / len(subset) if subset else None,
                           "predicted": dict(Counter(row.get("category") or "invalid" for row in subset))}
    correct = sum(row.get("category") == row["expected"] for row in rows)
    return {"total": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else 0,
            "by_class": by_class, "methods": dict(Counter(row.get("method", "error") for row in rows)),
            "game_missed_as_chat": sum(row["expected"] == GAME and row.get("category") == GENERAL for row in rows),
            "chat_sent_to_game": sum(row["expected"] == GENERAL and row.get("category") == GAME for row in rows)}
