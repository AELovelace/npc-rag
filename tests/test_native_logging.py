"""Regression for CUDA errors disappearing behind upstream Decider's quiet log callback."""
import gc
from types import SimpleNamespace

from npc_rag import native_logging


def test_native_warnings_errors_and_continuations_reach_stderr(monkeypatch):
    output = []
    monkeypatch.setattr(native_logging.os, "write", lambda fd, data: output.append((fd, data)))
    native_logging._forward(2, b"normal loading\n", None)
    native_logging._forward(5, b"normal continuation\n", None)
    native_logging._forward(4, b"CUDA error: actual reason\n", None)
    native_logging._forward(5, b"current device: 1\n", None)
    native_logging._forward(3, b"native warning\n", None)
    native_logging._forward(1, b"debug\n", None)
    assert output == [(2, b"CUDA error: actual reason\n"), (2, b"current device: 1\n"), (2, b"native warning\n")]


def test_callback_is_retained_and_can_be_reinstalled(monkeypatch):
    installed = []
    monkeypatch.setattr(native_logging, "_callback", None)
    llama = SimpleNamespace(llama_log_callback=lambda fn: fn, llama_log_set=lambda cb, _: installed.append(cb))
    native_logging.install_native_error_logging(llama)
    gc.collect()
    native_logging.install_native_error_logging(llama)
    assert installed == [native_logging._callback, native_logging._callback]


def test_closed_stderr_does_not_raise_through_native_callback(monkeypatch):
    def closed(*args):
        raise OSError("stderr closed")
    monkeypatch.setattr(native_logging.os, "write", closed)
    native_logging._forward(4, b"error during shutdown", None)
