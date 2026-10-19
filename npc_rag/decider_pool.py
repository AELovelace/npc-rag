"""One spawned Decider process per GPU, with a shared cancellation-safe admission queue."""
from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
import logging
import multiprocessing
import os
import threading
import time

from .decider_devices import detect_devices, pinned_device, select_devices
from .llm import LLMError

logger = logging.getLogger(__name__)
_native_backend = None  # Each spawned process owns a distinct model; no model crosses a pickle boundary.


def _watch_parent():
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def wait_for_exit():
        parent.join()
        os._exit(0)  # A forced launcher restart must release this child's VRAM even during native inference.

    threading.Thread(target=wait_for_exit, name="decider-parent-watch", daemon=True).start()


def _worker_start(settings, device):
    from .classifier_backend import DeciderClassifier
    global _native_backend
    _native_backend = DeciderClassifier(settings)
    with pinned_device(device["name"]) as actual:
        _native_backend._load()  # Reuse the qualified tokenizer, prompt, probabilities and model provisioning.
    began = time.perf_counter()
    _native_backend._decide("A quiet afternoon sounds pleasant.")  # Exercise decode before advertising readiness; no player data/cache.
    metadata = _native_backend.view()
    metadata["startup_inference_ms"] = round((time.perf_counter() - began) * 1000)
    return {"metadata": metadata, "device": actual, "pid": os.getpid()}


def _worker_decide(message):
    try:
        decision = _native_backend._decide(message)
        return str(decision), decision.game_probability  # A plain tuple preserves confidence across IPC.
    except Exception as exc:
        raise RuntimeError(f"Native Decider failed ({type(exc).__name__})") from None  # Keep player text out of IPC errors.


def _worker_stop():
    global _native_backend
    if _native_backend is not None:
        _native_backend._unload()
        _native_backend._worker.shutdown(wait=True)
        _native_backend = None


class ProcessWorker:
    """Keep at most one operation submitted to a single-process executor."""

    def __init__(self, device, *, operations=None):
        self.device = device
        self.operations = operations or (_worker_start, _worker_decide, _worker_stop)
        self.executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                            initializer=_watch_parent)
        self.pending = None
        self.healthy = False
        self.failures = self.completed = 0
        self.pid = None
        self.metadata = {}

    async def start(self, settings):
        self.pending = asyncio.get_running_loop().run_in_executor(self.executor, self.operations[0], settings, self.device)
        try:
            result = await asyncio.shield(self.pending)
        except BrokenProcessPool:
            raise RuntimeError(f"Decider worker {self.device['name']} exited during model loading/startup inference; "
                               "check the preceding native CUDA/ROCm/Vulkan error in the RAG log") from None
        self.metadata, self.device, self.pid = result["metadata"], result["device"], result["pid"]
        self.pending = None
        self.healthy = True

    def submit(self, message):
        self.pending = asyncio.get_running_loop().run_in_executor(self.executor, self.operations[1], message)
        return self.pending  # Pool owns admission; the executor never accumulates abandoned queued requests.

    async def close(self):
        if self.pending is not None:
            try:
                await asyncio.shield(self.pending)  # Caller timeouts cannot free an executing native context.
            except Exception:
                pass
        try:
            await asyncio.get_running_loop().run_in_executor(self.executor, self.operations[2])
        except BrokenProcessPool:
            pass  # A crashed process already released its native resources.
        finally:
            self.healthy = False
            await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)


async def _discover():
    executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_watch_parent)
    try:
        return await asyncio.shield(asyncio.get_running_loop().run_in_executor(executor, detect_devices))
    finally:
        await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)


class DeciderPool:
    """Dispatch to the next idle GPU; keep cache and wiki routing in the existing RAG service."""

    def __init__(self, settings, *, discover=None, worker_factory=ProcessWorker, result_factory=None):
        self.settings = settings
        self._discover, self._worker_factory = discover or _discover, worker_factory
        self._result_factory = result_factory
        self._workers = []
        self._available = deque()
        self._changed = asyncio.Event()
        self._started = self._closed = False
        self._queued = 0
        self._close_task = None

    async def start(self):
        if self._closed or self._started:
            raise RuntimeError("Decider pool cannot be started twice or after closing")
        try:
            devices = select_devices(await self._discover(), self.settings.decider_gpu_devices)
            # Children need model settings, never the cloud or service API credentials.
            child_settings = replace(self.settings, api_key="", llm_api_key="", openai_api_key="")
            for device in devices:
                worker = self._worker_factory(device)
                self._workers.append(worker)  # Include a partially loaded worker in failure cleanup.
                logger.info("Loading and testing Decider worker: %s (%s)", device["name"], device["description"])
                await worker.start(child_settings)
                if self._identity(worker) != self._identity(self._workers[0]):
                    raise RuntimeError("Decider worker models or routing policies disagree")
                self._available.append(worker)
                logger.info("Decider worker ready: %s (%s), pid=%s", device["name"], device["description"], worker.pid)
            self._started = True  # Readiness requires every selected GPU to load successfully.
        except BaseException:
            await self.aclose()
            raise

    @staticmethod
    def _identity(worker):
        view = worker.metadata
        return ("decider", view["model"], view["gguf_file"], view["decision_policy"])

    async def _take(self):
        while True:
            if self._closed or not self._started or not any(w.healthy for w in self._workers):
                raise LLMError("Decider pool is unavailable")
            if self._available:
                return self._available.popleft()
            self._changed.clear()  # No await between checking availability and registering this wait.
            self._queued += 1
            try:
                await self._changed.wait()
            finally:
                self._queued -= 1  # A cancelled waiter disappears without submitting native work.

    def _finished(self, worker, future):
        worker.pending = None
        error = future.exception() if not future.cancelled() else None
        if error is not None:
            worker.failures += 1  # Count native failures even when the caller already timed out.
            if isinstance(error, BrokenProcessPool):
                worker.healthy = False
        else:
            worker.completed += 1
        if worker.healthy and not self._closed:
            self._available.append(worker)
        self._changed.set()  # Wake queued callers, including when the last healthy worker has died.

    async def decide(self, message):
        from .classifier import ModelDecision
        worker = await self._take()
        try:
            pending = worker.submit(message)
        except Exception as exc:
            worker.failures += 1
            worker.healthy = False
            self._changed.set()
            raise LLMError(f"Decider worker unavailable ({type(exc).__name__})") from None
        pending.add_done_callback(lambda future: self._finished(worker, future))
        try:
            category, probability = await asyncio.shield(pending)
            return (self._result_factory or ModelDecision)(category, probability)
        except Exception as exc:
            raise LLMError(f"Decider inference failed ({type(exc).__name__})") from None

    async def health(self):
        return self._started and not self._closed and any(w.healthy for w in self._workers)

    async def model_identity(self):
        return self._identity(self._workers[0]) if await self.health() else None

    def view(self):
        base = dict(self._workers[0].metadata) if self._workers else {"backend": "decider"}
        workers = [{"device": w.device["name"], "description": w.device["description"],
                    "device_id": w.device.get("device_id"), "pid": w.pid, "healthy": w.healthy,
                    "startup_inference_ms": w.metadata.get("startup_inference_ms"),
                    "busy": w.pending is not None, "completed": w.completed, "failures": w.failures}
                   for w in self._workers]
        return {**base, "execution": "gpu-process-pool", "device_selection": self.settings.decider_gpu_devices,
                "capacity": len(workers), "healthy_capacity": sum(w["healthy"] for w in workers),
                "busy": any(w["busy"] for w in workers), "in_flight": sum(w["busy"] for w in workers),
                "queued": self._queued, "failures": sum(w["failures"] for w in workers), "workers": workers}

    async def aclose(self):
        if self._close_task is None:
            self._closed = True
            self._changed.set()  # Reject waiting/new requests immediately, then drain native operations.
            self._available.clear()
            self._close_task = asyncio.ensure_future(asyncio.gather(*(w.close() for w in self._workers)))
        await asyncio.shield(self._close_task)  # Repeated/cancelled cleanup callers share the same draining task.
