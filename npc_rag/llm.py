"""Async client for llama.cpp's OpenAI-compatible /v1/chat/completions endpoint."""
from __future__ import annotations

import asyncio
import re

import httpx

_THINK = re.compile(r"<think>.*?</think>|^.*?</think>", re.S)  # Qwen-style reasoning blocks, closed or orphaned.


class LLMError(RuntimeError):
    """llama.cpp was unreachable, timed out or answered with something unusable."""


class LlamaClient:
    def __init__(self, base_url: str, model: str = "local", api_key: str = "", concurrency: int = 2,
                 transport: httpx.AsyncBaseTransport | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}  # Only sent when llama.cpp has --api-key.
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, transport=transport)
        self._model = model
        self._slots = asyncio.Semaphore(max(1, concurrency))  # Never hold more llama.cpp slots than configured.

    async def chat(self, messages: list[dict], *, max_tokens: int, temperature: float, timeout: float) -> str:
        """Send one non-streaming chat request and return the assistant text with any reasoning removed."""
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},  # Qwen3 thinking would burn the token budget before replying.
        }
        try:
            async with self._slots:
                r = await self._client.post("/v1/chat/completions", json=body, timeout=timeout)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"llama.cpp request failed: {exc!r}") from exc
        text = _THINK.sub("", text).strip()  # Belt and braces if a template ignores enable_thinking.
        if not text:
            raise LLMError("llama.cpp returned an empty reply")
        return text

    async def health(self) -> bool:
        """True when llama.cpp answers /health (or /v1/models on older builds)."""
        for path in ("/health", "/v1/models"):
            try:
                r = await self._client.get(path, timeout=5)
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
                continue
        return False

    async def aclose(self) -> None:
        await self._client.aclose()
