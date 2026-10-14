"""Interchangeable HTTP chat classifier and local, single-worker Decider GGUF classifier."""
from __future__ import annotations

import asyncio
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .classifier import CLASSIFY_GRAMMAR, GAME, GENERAL, ModelDecision, classifier_messages, parse_verdict
from .config import ROOT
from .decider_policy import DECIDER_LABELS, DECIDER_OPTIONS, DECIDER_POLICY, DECIDER_QUESTION, policy_for_layout
from .decider_prompt import checkpoint_layout, create_routing_decider
from .llm import LlamaClient, LLMError
from .model_download import CUSTOM_GGUF, CUSTOM_MODEL, ensure_custom_decider


class ChatClassifier:
    """Preserve the existing llama.cpp request and strict label parser."""

    def __init__(self, client, timeout):
        self.client, self.timeout = client, timeout

    async def start(self):
        pass  # HTTP models are already loaded by their external server.

    async def decide(self, message):
        return parse_verdict(await self.client.chat(
            classifier_messages(message), grammar=CLASSIFY_GRAMMAR,
            max_tokens=4, temperature=0, timeout=self.timeout))

    async def model_identity(self):
        return await self.client.model_identity()

    async def health(self):
        return await self.client.health()

    def view(self):
        return {"backend": "llamacpp"}

    async def aclose(self):
        await self.client.aclose()


class DeciderClassifier:
    """Own one native context; cancellation never permits overlapping access to its buffers."""

    def __init__(self, settings, *, factory=None):
        self.settings, self._factory = settings, factory
        self._model = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="npc-decider")
        self._slot = asyncio.Lock()  # Callers queue here, not in the executor's unbounded work queue.
        self._closed = False
        self._busy = False
        self._failures = 0
        self._prompt_layout = 'state_first'
        self._model_source = settings.decider_model
        self._gguf_file = settings.decider_gguf_file

    async def _native(self, function):
        await self._slot.acquire()  # Cancellation while waiting prevents submission entirely.
        if self._closed:
            self._slot.release()
            raise LLMError("Decider is closed")
        loop = asyncio.get_running_loop()
        self._busy = True
        try:
            pending = loop.run_in_executor(self._worker, function)
        except BaseException:
            self._busy = False
            self._slot.release()
            raise

        def completed(future):
            self._busy = False
            self._slot.release()  # The native call owns this slot until it actually finishes.
            if not future.cancelled():
                future.exception()  # Consume late exceptions when the original caller timed out.

        pending.add_done_callback(completed)
        return await asyncio.shield(pending)  # Python cancellation cannot stop a running llama_decode.

    def _load(self):
        if self._model is not None:
            return
        if self._factory is None:
            if sys.version_info < (3, 11):
                raise RuntimeError("Decider requires Python 3.11 or newer")
            try:
                from decider.infer import Decider
                import llama_cpp  # Fail at startup if the optional native backend was not installed.
            except ImportError as exc:
                raise RuntimeError("Install requirements-decider.txt in the RAG environment before enabling Decider") from exc
            from .native_logging import install_native_error_logging
            install_native_error_logging(llama_cpp)
            factory = create_routing_decider
        else:
            factory = self._factory  # Tests substitute the native model without downloading any weights.
        source = self.settings.decider_model
        if self._factory is None and source == CUSTOM_MODEL:
            source = str(ensure_custom_decider())  # Fetch matching weights and sidecars before native loading.
            self._gguf_file = CUSTOM_GGUF  # Old .env filenames cannot select mismatched weights for this release.
        local = Path(source)
        if not local.is_absolute() and (ROOT / local).is_dir():
            source = str(ROOT / local)  # Local model folders resolve relative to the npc-rag checkout.
        if self._factory is None:
            from huggingface_hub import hf_hub_download
            config_path = (Path(source) / "decider_config.json" if Path(source).is_dir()
                           else Path(hf_hub_download(source, "decider_config.json")))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or "temperature" not in config:
                raise RuntimeError("Decider requires its matching decider_config.json with calibration temperature")
            self._prompt_layout = checkpoint_layout(config)
            # Upstream tolerates missing config; require it here so an offline/error path cannot lose calibration silently.
        layout_options = {'prompt_order': self._prompt_layout} if self._factory is None else {}
        log_options = {'verbose': True} if self._factory is None else {}  # Bypass upstream's all-level log silencer.
        self._model_source = source  # Health and decision-cache identity describe the actual local release.
        self._model = factory(source, **layout_options, gguf_file=self._gguf_file,
                              gguf_options={"n_gpu_layers": self.settings.decider_gpu_layers,
                                            "n_threads": self.settings.decider_threads,
                                            "n_ctx": self.settings.decider_context, "n_seq_max": 1, **log_options})

    async def start(self):
        await self._native(self._load)  # Load before readiness; first launch may download the model/tokenizer/config.

    def _decide(self, message):
        if self._model is None:
            raise RuntimeError("Decider has not been started")
        # Refuse overlong inputs instead of silently truncating away a trailing gameplay question.
        state = json.dumps({"message": message}, ensure_ascii=False)
        limit = self.settings.decider_context - 1024  # Reserve space for the question, options and answer slot.
        if len(self._model.m.tok.encode("Context:\n" + state, add_special_tokens=False)) > limit:
            raise ValueError("Message exceeds Decider context budget")
        rows = self._model.decide(state, [{"question": DECIDER_QUESTION, "options": list(DECIDER_OPTIONS)}],
                                  max_ctx_tokens=limit)
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError("Invalid Decider result")
        row = rows[0]
        probs = row["probs"]
        if set(probs) != set(DECIDER_OPTIONS):
            raise ValueError("Invalid Decider options")
        values = [float(probs[label]) for label in DECIDER_OPTIONS]
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values) or not math.isclose(sum(values), 1, abs_tol=1e-4):
            raise ValueError("Invalid Decider probabilities")
        choice = row["choice"]
        if choice not in probs or float(probs[choice]) < max(values):
            raise ValueError("Invalid Decider choice")
        return ModelDecision(DECIDER_LABELS[choice], values[0])  # Retain GAME confidence even when CHAT wins.

    async def decide(self, message):
        try:
            return await self._native(lambda: self._decide(message))
        except Exception as exc:
            self._failures += 1
            raise LLMError(f"Decider inference failed ({type(exc).__name__})") from None  # Never log player text.

    async def model_identity(self):
        if not await self.health():
            return None
        engine = getattr(self, '_engine_name', 'gguf')
        return ("decider", self._model_source,
                self._gguf_file if engine == 'gguf' else engine, policy_for_layout(self._prompt_layout))

    async def health(self):
        return self._model is not None and not self._closed  # Readiness, not a new inference on every health poll.

    def view(self):
        return {"backend": "decider", "decision_policy": policy_for_layout(self._prompt_layout),
                "prompt_layout": self._prompt_layout, "prefix_cache": False,
                "model": self._model_source,
                "release": CUSTOM_MODEL if self.settings.decider_model == CUSTOM_MODEL else None,
                "engine": getattr(self, '_engine_name', 'gguf'),
                "gguf_file": None if getattr(self, '_engine_name', 'gguf') == 'torch-bf16' else self._gguf_file,
                "threads": self.settings.decider_threads, "gpu_layers": self.settings.decider_gpu_layers,
                "context": self.settings.decider_context, "capacity": 1, "busy": self._busy,
                "failures": self._failures}

    async def aclose(self):
        if self._closed:
            return
        self._closed = True  # Stop new submissions before waiting for any uncancellable native call.
        async with self._slot:
            await asyncio.get_running_loop().run_in_executor(self._worker, self._unload)
        await asyncio.to_thread(self._worker.shutdown, wait=True, cancel_futures=True)

    def _unload(self):
        self._model = None  # Release native model/context on its owning worker thread.


def make_classifier(settings, *, client=None):
    """Use the same adapter in the service and CLI; injected HTTP clients keep existing tests compatible."""
    if client is not None:
        return ChatClassifier(client, settings.classifier_timeout)
    if settings.classifier_backend == "decider":
        if settings.decider_gpu_layers != 0:
            from .decider_pool import DeciderPool
            return DeciderPool(settings)  # GPU mode detects cards at startup and isolates one context per process.
        return DeciderClassifier(settings)
    return ChatClassifier(LlamaClient(settings.classifier_url, settings.classifier_model,
                                     settings.llm_api_key, settings.llm_concurrency), settings.classifier_timeout)


def make_torch_eval_classifier(settings, model_path):
    """Evaluate local merged weights through the serving policy without changing serving configuration."""
    from dataclasses import replace
    path = Path(model_path).resolve()
    config = json.loads((path / 'decider_config.json').read_text(encoding='utf-8'))
    if 'temperature' not in config:
        raise ValueError('Evaluation checkpoint lacks temperature')
    layout = checkpoint_layout(config)

    def factory(source, **kwargs):
        return create_routing_decider(source, prompt_order=layout, device='cuda', use_graphs=False)

    backend = DeciderClassifier(replace(settings, classifier_backend='decider', decider_model=str(path)), factory=factory)
    backend._engine_name = 'torch-bf16'
    backend._prompt_layout = layout
    return backend
