"""Async client for llama.cpp's OpenAI-compatible /v1/chat/completions endpoint."""
from __future__ import annotations

import asyncio
import re
import time
from urllib.parse import quote

import httpx

_THINK = re.compile(r"<think>.*?</think>|^.*?</think>", re.S)  # Qwen-style reasoning blocks, closed or orphaned.


class LLMError(RuntimeError):
    """llama.cpp was unreachable, timed out or answered with something unusable."""


class OpenAIClient:
    """Responses API adapter; the existing local RAG and guards still surround every reply."""

    def __init__(self, api_key: str, model: str = "gpt-6-luna", concurrency: int = 4,
                 transport: httpx.AsyncBaseTransport | None = None):
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is required for OpenAI overflow")
        self._model = model
        self._slots = asyncio.Semaphore(max(1, concurrency))  # Bound paid requests, including guard retries.
        self._client = httpx.AsyncClient(base_url="https://api.openai.com", transport=transport,
                                        headers={"Authorization": f"Bearer {api_key}"}, follow_redirects=False)

    async def chat(self, messages: list[dict], *, max_tokens: int, temperature: float, timeout: float) -> str:
        """Send the prepared prompt without llama.cpp-only options; never log keys, prompts or API error bodies."""
        body = {"model": self._model, "input": messages, "max_output_tokens": max(16, max_tokens),
                "reasoning": {"effort": "none"}, "store": False, "stream": False}
        async def request():
            async with self._slots:
                response = await self._client.post("/v1/responses", json=body, timeout=timeout)
                if response.status_code != 200:
                    raise LLMError(f"OpenAI request failed (HTTP {response.status_code})")  # No automatic billable retries.
                data = response.json()
            if data.get("status") != "completed":
                raise LLMError("OpenAI did not complete the reply")  # Partial responses must not enter game memory.
            parts = []
            for item in data.get("output", []):
                if item.get("type") != "message" or item.get("role") != "assistant":
                    continue
                for part in item.get("content", []):
                    if part.get("type") == "refusal":
                        return "I can't help with that request."  # A refusal is terminal, never a retry on another provider.
                    if part.get("type") == "output_text":
                        parts.append(part["text"])
            text = "".join(parts).strip()
            if not text:
                raise LLMError("OpenAI returned an empty reply")
            return text
        try:
            return await asyncio.wait_for(request(), timeout=timeout)  # Includes waiting for a free API slot.
        except (httpx.HTTPError, asyncio.TimeoutError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise LLMError("OpenAI request failed or timed out") from None  # The caller's logs never receive provider response data.

    async def health(self) -> bool:
        """Check model access without generating text or consuming generation tokens."""
        try:
            response = await self._client.get("/v1/models/" + quote(self._model, safe=""), timeout=5)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()  # Release the dedicated authenticated HTTPS connection pool.


class OverflowClient:
    """One 9092 process chooses local or cloud only after RAG has prepared the generation prompt."""

    def __init__(self, local, cloud, *, local_capacity=4, cloud_capacity=4, cloud_timeout=12, now=time.monotonic):
        self.local, self.cloud = local, cloud
        self._local_slots = asyncio.Semaphore(local_capacity)
        self._cloud_slots = asyncio.Semaphore(cloud_capacity)
        self._local_capacity, self._cloud_capacity = local_capacity, cloud_capacity
        self._local_busy = self._cloud_busy = 0
        self._cloud_timeout, self._now, self._cloud_down_until = cloud_timeout, now, 0

    async def _local(self, messages, **kwargs):
        async with self._local_slots:
            self._local_busy += 1  # Reserve before awaiting HTTP; simultaneous replies cannot steal the same slot.
            try:
                return await self.local.chat(messages, **kwargs)
            finally:
                self._local_busy -= 1

    async def _cloud(self, messages, **kwargs):
        async with self._cloud_slots:
            self._cloud_busy += 1
            try:
                return await self.cloud.chat(messages, **{**kwargs, "timeout": min(kwargs["timeout"], self._cloud_timeout)})
            except LLMError:
                self._cloud_down_until = self._now() + 30  # Rate limits/outages do not generate a paid retry storm.
                raise
            finally:
                self._cloud_busy -= 1

    def _cloud_available(self):
        return not self._cloud_slots.locked() and self._now() >= self._cloud_down_until

    async def chat(self, messages, **kwargs):
        """Background impressions and summaries count toward local capacity but never use paid overflow."""
        try:
            return await asyncio.wait_for(self._local(messages, **kwargs), timeout=kwargs["timeout"])
        except asyncio.TimeoutError:
            raise LLMError("Local model queue or generation timed out") from None

    async def reply(self, messages, **kwargs):
        """Choose at generation time; waiting and local failover share the caller's total deadline."""
        async def route():
            if self._local_slots.locked() and self._cloud_available():
                return await self._cloud(messages, **kwargs)
            try:
                return await self._local(messages, **kwargs)
            except LLMError:
                if self._cloud_available():
                    return await self._cloud(messages, **kwargs)  # Fast local failures can still get a cloud answer.
                raise
        try:
            return await asyncio.wait_for(route(), timeout=kwargs["timeout"])
        except asyncio.TimeoutError:
            raise LLMError("Reply queue or generation timed out") from None

    async def health(self):
        local_ok, cloud_ok = await asyncio.gather(self.local.health(), self.cloud.health())
        return local_ok or cloud_ok  # Keep the RAG node reachable if either generation backend works.

    def view(self):
        return {"enabled": True, "local_capacity": self._local_capacity, "local_in_flight": self._local_busy,
                "cloud_capacity": self._cloud_capacity, "cloud_in_flight": self._cloud_busy,
                "cloud_cooldown_s": max(0, round(self._cloud_down_until - self._now()))}  # No credentials or prompts.

    async def aclose(self):
        await asyncio.gather(self.local.aclose(), self.cloud.aclose())


def make_reply_llm(settings):
    """An API key enables overflow on the existing RAG service; no key preserves local-only generation."""
    if settings.openai_api_key and not settings.api_key:
        raise ValueError("NPC_API_KEY is required for OpenAI overflow")
    local = make_llm(settings.llm_url, settings.llm_model, settings.llm_api_key, settings.llm_concurrency)
    if not settings.openai_api_key:
        return local
    copies = max(1, len([url for url in settings.llm_url.split(",") if url.strip()]))
    capacity = min(settings.openai_local_capacity, max(1, settings.llm_concurrency) * copies)
    cloud = OpenAIClient(settings.openai_api_key, settings.openai_model, settings.openai_concurrency)
    return OverflowClient(local, cloud, local_capacity=capacity, cloud_capacity=settings.openai_concurrency,
                          cloud_timeout=settings.openai_timeout)


class LlamaClient:
    def __init__(self, base_url: str, model: str = "local", api_key: str = "", concurrency: int = 2,
                 transport: httpx.AsyncBaseTransport | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}  # Only sent when llama.cpp has --api-key.
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, transport=transport)
        self._model = model
        self._slots = asyncio.Semaphore(max(1, concurrency))  # Never hold more llama.cpp slots than configured.

    async def chat(self, messages: list[dict], *, max_tokens: int, temperature: float, timeout: float,
                   grammar: str | None = None) -> str:
        """Send one non-streaming chat request, optionally constraining classifier output with a grammar."""
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},  # Qwen3 thinking would burn the token budget before replying.
        }
        if grammar is not None:
            body["grammar"] = grammar  # Reply generation remains unconstrained unless explicitly requested.
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

    async def model_identity(self) -> str | None:
        """Read the advertised model ID so cached labels can be invalidated after a hot swap."""
        try:
            response = await self._client.get("/v1/models", timeout=2)
            response.raise_for_status()
            models = response.json()["data"]
            return "|".join(sorted(item["id"] for item in models)) or None
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return None  # Unknown identity disables cache reuse instead of trusting potentially stale decisions.

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
