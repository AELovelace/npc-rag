"""Pieces both services share: API-key check and the per-player conversation memory."""
from __future__ import annotations

import hmac
import time
from collections import deque

from fastapi import Header, HTTPException


def api_key_guard(expected: str):
    """FastAPI dependency: require 'Authorization: Bearer <key>' or 'X-Api-Key: <key>' when a key is configured."""

    async def check(authorization: str = Header(default=""), x_api_key: str = Header(default="")) -> None:
        if not expected:
            return  # No NPC_API_KEY set: open (only do this on a private network).
        supplied = x_api_key or (authorization[7:] if authorization.lower().startswith("bearer ") else "")
        if not hmac.compare_digest(supplied.encode(), expected.encode()):  # Constant-time comparison.
            raise HTTPException(status_code=401, detail="missing or wrong API key")

    return check


class Sessions:
    """Short rolling memory per player so follow-up questions make sense. In-process only; a restart forgets."""

    def __init__(self, max_messages: int, ttl_seconds: int, max_players: int = 5000):
        self.max_messages = max_messages      # user + NPC lines kept per player.
        self.ttl = ttl_seconds                # Silence after which a player's memory is dropped.
        self.max_players = max_players        # Hard cap so memory can't grow without bound.
        self._data: dict[str, tuple[float, deque]] = {}

    def get(self, player_id: str) -> list[dict]:
        entry = self._data.get(player_id)
        if not entry or time.monotonic() - entry[0] > self.ttl:
            self._data.pop(player_id, None)   # Expired memories are forgotten on read.
            return []
        return list(entry[1])

    def add(self, player_id: str, user: str, assistant: str) -> None:
        self._prune()
        _, turns = self._data.get(player_id, (0.0, deque(maxlen=self.max_messages)))
        turns.append({"role": "user", "content": user})
        turns.append({"role": "assistant", "content": assistant})
        self._data[player_id] = (time.monotonic(), turns)

    def clear(self, player_id: str) -> None:
        self._data.pop(player_id, None)

    def _prune(self) -> None:
        now = time.monotonic()
        for pid in [p for p, (t, _) in self._data.items() if now - t > self.ttl]:
            del self._data[pid]               # Drop everyone who has gone quiet.
        while len(self._data) >= self.max_players:
            del self._data[min(self._data, key=lambda p: self._data[p][0])]  # Then the oldest, if still full.
