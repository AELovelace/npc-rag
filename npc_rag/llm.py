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


class LlamaPool:
    """Several llama.cpp servers for one role (e.g. one model copy per graphics card): each call goes to the copy with
    the fewest requests in flight, and a copy that fails is skipped for that call. Same interface as LlamaClient."""

    def __init__(self, clients: list[LlamaClient]):
        self._clients = clients
        self._busy = [0] * len(clients)                      # Requests in flight per copy.

    async def chat(self, messages: list[dict], **kwargs) -> str:
        order = sorted(range(len(self._clients)), key=lambda i: self._busy[i])  # Least busy first.
        last: LLMError | None = None
        for i in order:
            self._busy[i] += 1
            try:
                return await self._clients[i].chat(messages, **kwargs)
            except LLMError as exc:
                last = exc                                     # Try the next copy; the caller only sees an error if all fail.
            finally:
                self._busy[i] -= 1
        raise last or LLMError("no llama.cpp servers configured")

    async def health(self) -> bool:
        return any(await asyncio.gather(*(c.health() for c in self._clients)))  # Up while any copy answers.

    async def aclose(self) -> None:
        await asyncio.gather(*(c.aclose() for c in self._clients))


def make_llm(urls: str, model: str = "local", api_key: str = "", concurrency: int = 2):
    """LLM_URL may list several servers ("http://127.0.0.1:9090,http://127.0.0.1:9091"); one URL keeps a plain client."""
    parts = [u.strip().rstrip("/") for u in str(urls).split(",") if u.strip()]
    if len(parts) <= 1:
        return LlamaClient(parts[0] if parts else urls, model, api_key, concurrency)
    return LlamaPool([LlamaClient(u, model, api_key, concurrency) for u in parts])  # Each copy gets its own slot limit.
