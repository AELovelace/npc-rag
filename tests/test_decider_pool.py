"""GPU discovery, real spawned-process IPC, and deterministic pool lifetime regressions."""
import asyncio
import ctypes
from dataclasses import replace
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from concurrent.futures.process import BrokenProcessPool

import httpx
import pytest
from fastapi.testclient import TestClient

from npc_rag.classifier import GENERAL, ModelDecisionCache
from npc_rag.classifier_backend import DeciderClassifier, make_classifier
from npc_rag.decider_devices import pinned_device, select_devices
from npc_rag.decider_pool import DeciderPool, ProcessWorker, _watch_parent
from npc_rag.llm import LLMError, LlamaClient
from npc_rag.service import create_app


def card(index, **extra):
    return {"name": f"CUDA{index}", "description": f"Test GPU {index}",
            "device_id": f"pci:{index}", "integrated": False, **extra}


def metadata(settings):
    return {"backend": "decider", "model": settings.decider_model, "gguf_file": settings.decider_gguf_file,
            "decision_policy": "game-chat-v3", "capacity": 1, "gpu_layers": settings.decider_gpu_layers}


class ControlledWorker:
    """Explicit futures let tests hold native work after a caller has cancelled."""

    def __init__(self, device):
        self.device, self.metadata = device, {}
        self.pending, self.pid = None, 123
        self.healthy = self.closed = False
        self.completed = self.failures = 0
        self.calls = []

    async def start(self, settings):
        assert not settings.api_key and not settings.llm_api_key and not settings.openai_api_key
        self.metadata, self.healthy = metadata(settings), True

    def submit(self, message):
        assert self.pending is None  # Overlap on one native context would corrupt model state.
        self.calls.append(message)
        self.pending = asyncio.get_running_loop().create_future()
        return self.pending

    async def close(self):
        if self.pending is not None:
            try:
                await asyncio.shield(self.pending)
            except Exception:
                pass
        self.closed, self.healthy = True, False


async def ready(predicate):
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("Expected async state transition did not occur")


def pool_for(settings, count=2, factory=ControlledWorker):
    async def discover():
        return [card(i) for i in range(count)]
    return DeciderPool(replace(settings, decider_gpu_layers=-1), discover=discover, worker_factory=factory)


def test_auto_selection_deduplicates_cards_and_manual_selection_can_include_integrated():
    rows = [card(0), card(1), card(9, name="Vulkan0", device_id="pci:0"), card(2, integrated=True)]
    assert [r["name"] for r in select_devices(rows, "auto")] == ["CUDA0", "CUDA1"]
    assert [r["name"] for r in select_devices(rows, "CUDA2,CUDA0")] == ["CUDA2", "CUDA0"]
    with pytest.raises(RuntimeError, match="same physical"):
        select_devices(rows, "CUDA0,Vulkan0")
    with pytest.raises(RuntimeError, match="Unknown Decider"):
        select_devices(rows, "CUDA8")
    with pytest.raises(RuntimeError, match="No dedicated"):
        select_devices([], "auto")


def test_native_pinning_restricts_device_list_and_restores_loader_on_error(monkeypatch):
    def original(path, params):
        pointers = ctypes.cast(params.devices, ctypes.POINTER(ctypes.c_void_p))
        assert pointers[0] == 42 and pointers[1] is None
        assert params.split_mode == 0 and params.main_gpu == 0
        raise RuntimeError("test allocation failure")
    llama = SimpleNamespace(llama_model_load_from_file=original, LLAMA_SPLIT_MODE_NONE=0)
    registry = SimpleNamespace(llama=llama, rows=lambda: [card(1, pointer=42)])
    monkeypatch.setattr("npc_rag.decider_devices.NativeDevices", lambda: registry)
    with pytest.raises(RuntimeError, match="allocation"):
        with pinned_device("CUDA1") as device:
            assert "pointer" not in device
            llama.llama_model_load_from_file("model.gguf", SimpleNamespace())
    assert llama.llama_model_load_from_file is original
    with pytest.raises(RuntimeError, match="disappeared"):
        with pinned_device("CUDA9"):
            pass


@pytest.mark.parametrize("value", ["", "CUDA0,", "CUDA0,CUDA0", "auto,CUDA0"])
def test_invalid_device_selection(settings, value):
    with pytest.raises(ValueError, match="DECIDER_GPU_DEVICES"):
        replace(settings, decider_gpu_devices=value)


def test_device_configuration_defaults_and_override(settings, monkeypatch, tmp_path):
    from npc_rag.config import load_settings
    monkeypatch.delenv("DECIDER_GPU_DEVICES", raising=False)
    assert load_settings(str(tmp_path / "missing.env")).decider_gpu_devices == "auto"
    monkeypatch.setenv("DECIDER_GPU_DEVICES", " Vulkan1, Vulkan0 ")
    assert load_settings(str(tmp_path / "missing.env")).decider_gpu_devices == "Vulkan1,Vulkan0"
    assert isinstance(make_classifier(replace(settings, classifier_backend="decider", decider_gpu_layers=-1)), DeciderPool)
    cpu = make_classifier(replace(settings, classifier_backend="decider", decider_gpu_layers=0))
    assert isinstance(cpu, DeciderClassifier)
    asyncio.run(cpu.aclose())


@pytest.mark.parametrize("count", [1, 2, 4])
def test_parallel_workers_dispatch_next_request_to_first_free_card(settings, count):
    async def run():
        pool = pool_for(settings, count)
        await pool.start()
        tasks = [asyncio.create_task(pool.decide(str(i))) for i in range(count + 1)]
        await ready(lambda: pool.view()["queued"] == 1)
        assert pool.view()["in_flight"] == count
        workers = pool._workers
        last = workers[-1]
        last.pending.set_result((GENERAL, .12))
        await ready(lambda: len(last.calls) == 2)
        assert last.calls == [str(count - 1), str(count)]
        for worker in workers:
            worker.pending.set_result((GENERAL, .12))
        results = await asyncio.gather(*tasks)
        assert all(r == GENERAL and r.game_probability == .12 for r in results)
        assert sum(w.completed for w in workers) == count + 1
        await pool.aclose()
        assert not await pool.health() and all(w.closed for w in workers)
    asyncio.run(run())


def test_cancelled_call_holds_worker_and_cancelled_waiters_never_submit(settings):
    async def run():
        pool = pool_for(settings)
        await pool.start()
        first = asyncio.create_task(pool.decide("cancelled caller"))
        second = asyncio.create_task(pool.decide("other card"))
        await ready(lambda: pool.view()["in_flight"] == 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert pool.view()["in_flight"] == 2
        outcomes = await asyncio.gather(*(asyncio.wait_for(pool.decide(str(i)), .01) for i in range(100)),
                                        return_exceptions=True)
        assert all(isinstance(r, TimeoutError) for r in outcomes)
        assert pool.view()["queued"] == 0
        assert sum(len(w.calls) for w in pool._workers) == 2
        pool._workers[0].pending.set_exception(RuntimeError("late native failure"))
        pool._workers[1].pending.set_result((GENERAL, .1))
        await second
        await ready(lambda: pool.view()["in_flight"] == 0)
        assert pool.view()["failures"] == 1 and await pool.health()
        next_call = asyncio.create_task(pool.decide("recovery"))
        await ready(lambda: pool.view()["in_flight"] == 1)
        next(w for w in pool._workers if w.pending is not None).pending.set_result((GENERAL, .1))
        assert await next_call == GENERAL
        await pool.aclose()
    asyncio.run(run())


def test_crashed_worker_is_removed_and_last_crash_wakes_waiters(settings):
    async def run():
        pool = pool_for(settings)
        await pool.start()
        tasks = [asyncio.create_task(pool.decide(str(i))) for i in range(3)]
        await ready(lambda: pool.view()["queued"] == 1)
        pool._workers[0].pending.set_exception(BrokenProcessPool("first worker died"))
        with pytest.raises(LLMError):
            await tasks[0]
        assert pool.view()["healthy_capacity"] == 1 and await pool.health()
        pool._workers[1].pending.set_exception(BrokenProcessPool("second worker died"))
        outcomes = await asyncio.gather(*tasks[1:], return_exceptions=True)
        assert all(isinstance(result, LLMError) for result in outcomes)
        assert not await pool.health() and await pool.model_identity() is None
        await pool.aclose()
    asyncio.run(run())


def test_shutdown_rejects_queue_but_drains_running_inference(settings):
    async def run():
        pool = pool_for(settings, 1)
        await pool.start()
        active = asyncio.create_task(pool.decide("active"))
        queued = asyncio.create_task(pool.decide("queued"))
        await ready(lambda: pool.view()["queued"] == 1)
        close = asyncio.create_task(pool.aclose())
        with pytest.raises(LLMError):
            await queued
        assert not close.done() and not await pool.health()
        pool._workers[0].pending.set_result((GENERAL, .1))
        assert await active == GENERAL
        await close
        await pool.aclose()
        assert pool._workers[0].calls == ["active"] and pool._workers[0].closed
    asyncio.run(run())


def test_partial_startup_failure_closes_every_created_worker(settings):
    class FailedWorker(ControlledWorker):
        async def start(self, config):
            if self.device["name"] == "CUDA1":
                raise RuntimeError("model allocation failed")
            await super().start(config)
    async def run():
        pool = pool_for(settings, factory=FailedWorker)
        with pytest.raises(RuntimeError, match="allocation"):
            await pool.start()
        assert len(pool._workers) == 2 and all(w.closed for w in pool._workers)
        assert not await pool.health()
        await pool.aclose()
    asyncio.run(run())


# Top-level functions are importable by Windows spawn: this exercises real IPC without native model downloads.
def spawned_start(settings, device):
    assert not settings.openai_api_key
    return {"metadata": metadata(settings), "device": device, "pid": os.getpid()}


def spawned_decide(message):
    if message == "fail":
        raise RuntimeError("synthetic inference failure")
    return GENERAL, .125


def spawned_stop():
    return None


def spawned_crash_start(settings, device):
    os._exit(17)  # Reproduce a native abort after weights load but before readiness.


def test_native_abort_during_startup_never_marks_pool_ready(settings):
    async def run():
        factory = lambda device: ProcessWorker(device, operations=(spawned_crash_start, spawned_decide, spawned_stop))
        pool = pool_for(settings, factory=factory)
        with pytest.raises(RuntimeError, match="CUDA0.*startup inference"):
            await pool.start()
        assert not await pool.health()
        assert all(not w.healthy and w.executor._shutdown_thread for w in pool._workers)
        await pool.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("fails", [False, True])
def test_worker_runs_real_decision_path_before_readiness(settings, monkeypatch, fails):
    from npc_rag import decider_pool
    calls = []

    class Backend:
        def __init__(self, config):
            self._worker = SimpleNamespace(shutdown=lambda **kwargs: None)

        def _load(self):
            calls.append("load")

        def _decide(self, message):
            assert message == "A quiet afternoon sounds pleasant."
            calls.append("decode")
            if fails:
                raise RuntimeError("native inference failed")

        def view(self):
            calls.append("ready")
            return metadata(settings)

        def _unload(self):
            calls.append("unload")

    monkeypatch.setattr("npc_rag.classifier_backend.DeciderClassifier", Backend)
    monkeypatch.setattr(decider_pool, "pinned_device", lambda name: nullcontext(card(0)))
    monkeypatch.setattr(decider_pool, "_native_backend", None)
    try:
        if fails:
            with pytest.raises(RuntimeError, match="inference failed"):
                decider_pool._worker_start(settings, card(0))
            assert calls == ["load", "decode"]
        else:
            result = decider_pool._worker_start(settings, card(0))
            assert calls == ["load", "decode", "ready"]
            assert result["metadata"]["startup_inference_ms"] >= 0
    finally:
        decider_pool._worker_stop()
    assert calls[-1] == "unload"


def orphan_parent(connection):
    executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_watch_parent)
    connection.send(executor.submit(os.getpid).result(timeout=10))
    connection.recv()  # Test opens the child handle before simulating the launcher's forced parent termination.
    os._exit(0)


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher process-lifetime regression")
def test_worker_exits_when_parent_is_forcibly_stopped():
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    context = multiprocessing.get_context("spawn")
    connection, child_connection = context.Pipe()
    parent = context.Process(target=orphan_parent, args=(child_connection,))
    handle = None
    parent.start()
    try:
        assert connection.poll(15), "Spawned worker never became ready"
        child_pid = connection.recv()
        handle = kernel.OpenProcess(0x00100000, False, child_pid)  # SYNCHRONIZE only; no unrelated process termination.
        assert handle
        connection.send("exit")
        parent.join(10)
        assert not parent.is_alive()
        assert kernel.WaitForSingleObject(handle, 10000) == 0, "Worker remained alive after parent exit"
    finally:
        if parent.is_alive():
            parent.terminate()
            parent.join(10)
        if handle:
            kernel.CloseHandle(handle)
        connection.close()
        child_connection.close()


def test_real_spawned_workers_preserve_confidence_cache_and_process_isolation(settings):
    async def run():
        factory = lambda device: ProcessWorker(device, operations=(spawned_start, spawned_decide, spawned_stop))
        pool = pool_for(replace(settings, openai_api_key="test-secret"), factory=factory)
        await pool.start()
        try:
            pids = {w.pid for w in pool._workers}
            assert len(pids) == 2 and os.getpid() not in pids
            rows = await asyncio.gather(pool.decide("a"), pool.decide("b"))
            assert all(r == GENERAL and r.game_probability == .125 for r in rows)
            assert [w.completed for w in pool._workers] == [1, 1]
            cache = ModelDecisionCache()
            first = await cache.get("same", lambda: pool.decide("same"))
            assert await cache.get("same", lambda: pool.decide("same")) is first
            with pytest.raises(LLMError):
                await pool.decide("fail")
            assert await pool.decide("recovered") == GENERAL
        finally:
            await pool.aclose()
        assert all(w.executor._shutdown_thread for w in pool._workers)
    asyncio.run(run())


def test_service_keeps_same_endpoint_and_shared_cache_with_gpu_pool(settings, embedder, index, monkeypatch):
    factory = lambda device: ProcessWorker(device, operations=(spawned_start, spawned_decide, spawned_stop))
    configured = replace(settings, classifier_backend="decider", decider_gpu_layers=-1, rag_game_evidence=2)
    pool = pool_for(configured, factory=factory)
    monkeypatch.setattr("npc_rag.service.make_classifier", lambda *a, **kw: pool)
    llm = LlamaClient(settings.llm_url, transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    app = create_app(configured, embedder=embedder, index=index, llm=llm)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["classifier"] and health["classifier_url"] is None
        assert health["classification"]["capacity"] == 2
        for _ in range(2):
            answer = client.post("/v1/classify", json={"message": "Tell me a joke."}).json()
            assert answer["category"] == GENERAL and answer["method"] == "llm"
            assert answer["scores"]["game_probability"] == .125
        assert sum(w.completed for w in pool._workers) == 1
    assert all(not w.healthy for w in pool._workers)
