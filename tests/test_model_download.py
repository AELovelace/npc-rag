"""Exercise startup installation using tiny HTTP fixtures, never production weights."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import httpx
import pytest

from npc_rag import classifier_backend, model_download as download


def _spec(content):
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


@pytest.fixture
def release(tmp_path, monkeypatch):
    configs = download._repository_configs()
    folder = tmp_path / "checkout"
    folder.mkdir()
    for name, content in configs.items():
        (folder / name).write_bytes(content)
    files = {download.CUSTOM_GGUF: b"GGUF tiny test weights", "tokenizer.json": b'{"version":"test"}',
             "manifest.json": b'{}', **configs}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, content in files.items():
            if name != download.CUSTOM_GGUF:
                bundle.writestr(name, content)
    payloads = {"/model": files[download.CUSTOM_GGUF], "/support": archive.getvalue()}
    spec = {"files": {name: _spec(content) for name, content in files.items()},
            "support_archive": _spec(payloads["/support"]),
            "download_url": "https://release.test/model", "support_url": "https://release.test/support"}
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps(spec))
    monkeypatch.setattr(download, "RELEASE_MANIFEST", manifest)
    monkeypatch.setattr(download, "CONFIG_DIRECTORY", folder)
    calls = []

    def respond(request):
        calls.append(request.url.path)
        assert "authorization" not in request.headers and "x-api-key" not in request.headers
        return httpx.Response(200, content=payloads[request.url.path])

    client = httpx.Client
    monkeypatch.setattr(download.httpx, "Client", lambda **kwargs: client(
        transport=httpx.MockTransport(respond), **kwargs))
    return SimpleNamespace(destination=tmp_path / "cache", files=files, payloads=payloads,
                           calls=calls, configs=folder, manifest=manifest, spec=spec)


def test_first_install_and_verified_offline_restart(release, monkeypatch):
    result = download.ensure_custom_decider(release.destination)
    assert result == release.destination
    assert release.calls == ["/support", "/model"]
    for name, content in release.files.items():
        assert (result / name).read_bytes() == content
    monkeypatch.setattr(download.httpx, "Client", lambda **kw: pytest.fail("Offline restart opened HTTP"))
    assert download.ensure_custom_decider(result) == result


def test_separate_release_filename_uses_same_verified_installer(release):
    filename = 'decider-chat-0-1.gguf'
    files = {filename if name == download.CUSTOM_GGUF else name: data for name, data in release.files.items()}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as bundle:
        for name, content in files.items():
            if name != filename:
                bundle.writestr(name, content)
    release.payloads['/support'] = archive.getvalue()
    spec = {**release.spec, 'files': {name: _spec(data) for name, data in files.items()},
            'support_archive': _spec(archive.getvalue())}
    release.manifest.write_text(json.dumps(spec))
    result = download.ensure_custom_decider(release.destination, release_manifest=release.manifest,
              config_loader=download._repository_configs, gguf_file=filename, model_name='decider-chat-0-1')
    assert (result / filename).read_bytes() == files[filename]
    assert not (result / download.CUSTOM_GGUF).exists()


def test_git_config_update_applies_without_asset_download(release):
    download.ensure_custom_decider(release.destination)
    release.calls.clear()
    weights = release.destination / download.CUSTOM_GGUF
    modified = weights.stat().st_mtime_ns
    config = json.loads((release.configs / "decider_config.json").read_bytes())
    config["temperature"] = .75
    updated = json.dumps(config).encode()
    (release.configs / "decider_config.json").write_bytes(updated)
    download.ensure_custom_decider(release.destination)
    assert not release.calls
    assert (release.destination / "decider_config.json").read_bytes() == updated
    assert weights.stat().st_mtime_ns == modified


def test_asset_repair_cannot_restore_old_hosted_config(release):
    download.ensure_custom_decider(release.destination)
    release.calls.clear()
    config_path = release.configs / "decider_config.json"
    config = json.loads(config_path.read_bytes())
    config["temperature"] = .8
    config_path.write_text(json.dumps(config))
    (release.destination / "tokenizer.json").write_bytes(b"broken")
    download.ensure_custom_decider(release.destination)
    assert release.calls == ["/support"]
    assert (release.destination / "decider_config.json").read_bytes() == config_path.read_bytes()


@pytest.mark.parametrize("payload", [b"truncated", b"X" * len(b"GGUF tiny test weights"), b"X" * 100])
def test_bad_download_never_publishes_partial_release(release, payload):
    release.payloads["/model"] = payload
    with pytest.raises(RuntimeError, match="startup check failed"):
        download.ensure_custom_decider(release.destination)
    assert not list(release.destination.iterdir())
    release.payloads["/model"] = release.files[download.CUSTOM_GGUF]
    download.ensure_custom_decider(release.destination)  # A later launch recovers without manual cleanup.


def test_same_size_local_corruption_is_repaired(release):
    download.ensure_custom_decider(release.destination)
    release.calls.clear()
    weights = release.destination / download.CUSTOM_GGUF
    weights.write_bytes(b"X" * weights.stat().st_size)
    download.ensure_custom_decider(release.destination)
    assert release.calls == ["/model"]
    assert weights.read_bytes() == release.files[download.CUSTOM_GGUF]


def test_concurrent_startups_share_install(release):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(download.ensure_custom_decider, [release.destination] * 2))
    assert results == [release.destination] * 2
    assert sorted(release.calls) == ["/model", "/support"]


def test_wrong_prompt_layout_fails_before_download(release):
    config_path = release.configs / "decider_config.json"
    config = json.loads(config_path.read_bytes())
    config.update(schema_first=True, routing_policy="game-chat-v4-schema-first")
    config_path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="state-first"):
        download.ensure_custom_decider(release.destination)
    assert not release.calls


def test_unexpected_archive_path_is_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape", b"bad")
    with pytest.raises(RuntimeError, match="unsafe path"):
        download._unpack(archive, tmp_path, {"../escape": _spec(b"bad")})
    assert not (tmp_path.parent / "escape").exists()


def test_custom_backend_uses_checked_local_files_and_reports_identity(settings, release, monkeypatch):
    monkeypatch.setattr("npc_rag.native_logging._callback", None)  # Restore the process callback reference after this fake library.
    monkeypatch.setitem(sys.modules, "decider.infer", SimpleNamespace(Decider=object))
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(
        llama_log_callback=lambda callback: callback, llama_log_set=lambda *args: None))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        hf_hub_download=lambda *a: pytest.fail("Custom release must not use Hugging Face")))
    monkeypatch.setattr(classifier_backend, "create_routing_decider", lambda source, **kw: SimpleNamespace(
        source=source, options=kw))
    monkeypatch.setattr(classifier_backend, "ensure_custom_decider",
                        lambda: download.ensure_custom_decider(release.destination))

    async def run():
        backend = classifier_backend.DeciderClassifier(replace(settings, decider_model=download.CUSTOM_MODEL))
        try:
            await backend.start()
            assert await backend.health()
            assert backend._model.source == str(release.destination)
            assert backend._model.options["gguf_file"] == download.CUSTOM_GGUF  # Overrides a legacy .env filename.
            assert backend._model.options["gguf_options"]["verbose"] is True  # Do not let upstream silence native errors.
            assert backend.view()["release"] == download.CUSTOM_MODEL
            assert (await backend.model_identity())[1:3] == (str(release.destination), download.CUSTOM_GGUF)
        finally:
            await backend.aclose()
    asyncio.run(run())
