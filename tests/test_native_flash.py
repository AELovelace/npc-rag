import ctypes
from types import SimpleNamespace

import pytest

from npc_rag import native_flash
from npc_rag.config import load_settings


class NewParams(ctypes.Structure):
    _fields_ = [("n_ctx", ctypes.c_uint32), ("flash_attn_type", ctypes.c_int)]


class OldParams(ctypes.Structure):
    _fields_ = [("n_ctx", ctypes.c_uint32), ("flash_attn", ctypes.c_bool)]


def fake_llama(params):
    inner = SimpleNamespace(llama_context_default_params=lambda: params(flash_attn_type=-1) if params is NewParams else params())
    return SimpleNamespace(llama_cpp=inner, llama_context_default_params=inner.llama_context_default_params)


def test_off_reaches_the_defaults_engines_read_and_auto_restores_them(monkeypatch):
    monkeypatch.setattr(native_flash, "_original", None)
    llama = fake_llama(NewParams)
    original = llama.llama_cpp.llama_context_default_params
    native_flash.set_flash_attention(llama, "off")
    assert llama.llama_context_default_params().flash_attn_type == 0
    assert llama.llama_cpp.llama_context_default_params().flash_attn_type == 0, "GGUFEngine may hold either module"
    native_flash.set_flash_attention(llama, "auto")
    assert llama.llama_cpp.llama_context_default_params is original


def test_older_llama_cpp_gets_its_boolean_flag(monkeypatch):
    monkeypatch.setattr(native_flash, "_original", None)
    llama = fake_llama(OldParams)
    native_flash.set_flash_attention(llama, "on")
    assert llama.llama_context_default_params().flash_attn is True


def test_decider_flash_attn_setting_is_validated(monkeypatch):
    monkeypatch.setenv("DECIDER_FLASH_ATTN", "OFF")
    assert load_settings().decider_flash_attn == "off"
    monkeypatch.setenv("DECIDER_FLASH_ATTN", "maybe")
    with pytest.raises(ValueError, match="DECIDER_FLASH_ATTN"):
        load_settings()
