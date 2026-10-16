"""Shared task admission, task identity, release integrity and service lifecycle."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from npc_rag.classifier_backend import make_classifier
from npc_rag.decider_pool import DeciderPool
from npc_rag.llm import LlamaClient
from npc_rag.multitask import (MODEL, GGUF, RELEASE_MANIFEST, MultitaskClassifier, TaskView,
                               repository_configs, validate_checkpoint)
from npc_rag.participation import make_participation, warmup_state
from npc_rag.service import create_app


@pytest.mark.parametrize('model', [MODEL, 'lidoll/decider-multitask-v2'])
def test_managed_alias_uses_its_own_pins_configs_and_cache(model, monkeypatch, settings):
    from npc_rag.multitask import RELEASES, ensure_multitask, is_multitask_source
    captured = {}
    def install(destination, **kwargs):
        captured.update(kwargs)
        return Path('fake-cache')
    monkeypatch.setattr('npc_rag.multitask.ensure_custom_decider', install)
    assert ensure_multitask(model=model) == Path('fake-cache')
    assert is_multitask_source(model)
    assert captured['model_name'] == model.split('/')[-1]
    assert captured['gguf_file'] == model.split('/')[-1] + '.gguf'
    assert captured['release_manifest'] == RELEASES[model][0]
    assert captured['config_loader']() == repository_configs(model)
    configured = replace(settings, classifier_backend='decider', decider_model=model,
                         decider_gpu_layers=0, participation_model='shared')
    routing = make_classifier(configured)
    assert routing.view()['release'] == model
    assert make_participation(configured, classifier=routing).backend is routing.backend
    other = MODEL if model != MODEL else 'lidoll/decider-multitask-v2'
    with pytest.raises(ValueError, match='must match'):
        make_participation(replace(configured, participation_model=other), classifier=routing)
    asyncio.run(routing.aclose())


def test_checked_in_release_and_prompt_schemas_match():
    configs = repository_configs()
    pin = json.loads(RELEASE_MANIFEST.read_text())
    assert pin['download_url'].endswith('/' + GGUF)
    assert pin['files'][GGUF]['bytes'] == 1274396192
    assert set(configs) <= set(pin['files'])  # Git may normalize line endings; runtime configs intentionally follow Git.
    config, schema = json.loads(configs['decider_config.json']), json.loads(configs['routing_policy.json'])
    schema['tasks']['participation']['prompt_layout'] = 'state_first'
    with pytest.raises(ValueError, match='schema'):
        validate_checkpoint(config, schema)


@pytest.mark.parametrize('model', [MODEL, 'lidoll/decider-multitask-v2'])
def test_factories_share_one_backend_and_reject_wrong_routing(settings, model):
    async def run():
        for layers in (0, -1):
            configured = replace(settings, classifier_backend='decider', decider_model=model,
                                 participation_model='shared', decider_gpu_layers=layers)
            routing = make_classifier(configured)
            participation = make_participation(configured, classifier=routing)
            assert isinstance(routing, TaskView)
            assert participation.backend is routing.backend
            assert isinstance(routing.backend, MultitaskClassifier if layers == 0 else DeciderPool)
            await participation.aclose()
            assert not routing.backend._closed
            await routing.aclose()
        with pytest.raises(ValueError, match='requires'):
            make_participation(replace(settings, participation_model='shared'))
    asyncio.run(run())


def test_tasks_share_busy_capacity_and_cancelled_waiters_do_not_submit(settings):
    from test_decider_pool import ControlledWorker, card, ready
    from npc_rag.multitask import _result

    async def run():
        async def discover():
            return [card(0)]
        pool = DeciderPool(settings, discover=discover, worker_factory=ControlledWorker, result_factory=_result)
        routing, participation = TaskView(pool, 'routing'), TaskView(pool, 'participation')
        await routing.start()
        await participation.start()
        worker = pool._workers[0]
        first = asyncio.create_task(routing.decide('Hello'))
        await ready(lambda: worker.pending is not None)
        queued = asyncio.create_task(participation.decide(warmup_state()))
        await ready(lambda: pool.view()['queued'] == 1)
        assert routing.view()['workers'] == participation.view()['workers']
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert pool.view()['in_flight'] == 1
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert len(worker.calls) == 1 and pool.view()['queued'] == 0
        worker.pending.set_result(('general', .1))
        await ready(lambda: pool.view()['in_flight'] == 0)
        next_call = asyncio.create_task(participation.decide(warmup_state()))
        await ready(lambda: worker.pending is not None)
        assert worker.calls[-1] == ('participation', warmup_state())
        worker.pending.set_result(('join', .9))
        decision = await next_call
        assert decision.label == 'join' and decision.join_probability == .9
        assert (await routing.model_identity()) != (await participation.model_identity())
        await participation.aclose()
        assert await routing.health()
        await routing.aclose()
        assert worker.closed
    asyncio.run(run())


def test_service_starts_and_closes_shared_model_once(settings, embedder, index, monkeypatch):
    class Backend:
        def __init__(self):
            self.starts = self.stops = 0
            self.requests = []
        async def start(self):
            self.starts += 1
        async def aclose(self):
            self.stops += 1
        async def health(self):
            return self.starts == 1 and self.stops == 0
        async def model_identity(self):
            return ('decider', MODEL, GGUF, 'multitask')
        def view(self):
            return {'capacity': 1, 'shared_pool': True, 'decision_policy': 'multitask', 'release': MODEL,
                    'workers': [{'pid': 123, 'healthy': True}]}
        async def decide(self, request):
            self.requests.append(request)
            return ('join', .9) if request[0] == 'participation' else ('general', .1)
    backend = Backend()
    monkeypatch.setattr('npc_rag.service.make_classifier', lambda *args, **kwargs: TaskView(backend, 'routing'))
    settings = replace(settings, classifier_backend='decider', decider_model=MODEL,
                       participation_enabled=True, participation_model='shared')
    llm = LlamaClient(settings.llm_url, transport=httpx.MockTransport(lambda req: httpx.Response(200, json={})))
    with TestClient(create_app(settings, embedder=embedder, index=index, llm=llm)) as client:
        response = client.post('/v1/participation/decide', json=warmup_state())
        assert response.status_code == 200 and response.json()['join'] is True
        assert response.json()['model'] == MODEL
        health = client.get('/health').json()
        assert health['classification']['workers'] == health['participation']['workers']
        assert backend.starts == 1 and backend.stops == 0
    assert backend.stops == 1


def test_archive_contents_match_installer_pins(tmp_path, monkeypatch):
    from npc_rag.model_download import _unpack, _matches
    release = Path(__file__).resolve().parents[1] / 'runs/releases/decider-multitask-v3-q4km'
    archive = release / 'decider-multitask-v3-support.zip'
    if not archive.is_file():
        pytest.skip('Local upload artifacts are not part of Git')
    pin = json.loads(RELEASE_MANIFEST.read_text())
    assert _matches(archive, pin['support_archive'])
    _unpack(archive, tmp_path, {name: spec for name, spec in pin['files'].items() if name != GGUF})
    assert (tmp_path / 'HOSTING.txt').is_file()
    from npc_rag.multitask import ensure_multitask
    def no_network(*args, **kwargs):
        raise AssertionError('A complete release must be reusable offline')
    monkeypatch.setattr('npc_rag.model_download.httpx.Client', no_network)
    assert ensure_multitask(release) == release.resolve()  # Verify the GGUF hash and every sidecar without a download.
