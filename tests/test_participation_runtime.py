"""Opt-in participation endpoint, task separation and native-result validation."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from npc_rag.participation import (ParticipationClassifier, ParticipationDecision, make_participation,
                                   repository_configs, warmup_state)
from npc_rag.participation_policy import OPTIONS, validate_checkpoint
from npc_rag.decider_pool import DeciderPool
from npc_rag.llm import LlamaClient
from npc_rag.service import create_app
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))


class Backend:
    def __init__(self, delay=0):
        self.delay, self.calls, self.closed = delay, [], False
    async def start(self):
        pass
    async def aclose(self):
        self.closed = True
    async def health(self):
        return True
    def view(self):
        return {'decision_policy': 'participation-sociable-v1-schema_first', 'capacity': 2}
    async def decide(self, state):
        self.calls.append(state)
        await asyncio.sleep(self.delay)
        return ParticipationDecision('join', .8)


def app_for(settings, embedder, index, backend):
    llm = LlamaClient(settings.llm_url, transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    classifier = LlamaClient(settings.classifier_url, transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    return create_app(settings, embedder=embedder, index=index, llm=llm, classifier_llm=classifier, participation_backend=backend)


def test_authenticated_endpoint_validates_state_and_returns_join_score(settings, embedder, index):
    backend = Backend()
    settings = replace(settings, participation_enabled=True, api_key='fixture-key')
    with TestClient(app_for(settings, embedder, index, backend)) as client:
        assert client.post('/v1/participation/decide', json=warmup_state()).status_code == 401
        headers = {'X-Api-Key': 'fixture-key'}
        assert client.post('/v1/participation/decide', json={**warmup_state(), 'gold': 'join'}, headers=headers).status_code == 422
        response = client.post('/v1/participation/decide', json=warmup_state(), headers=headers)
        assert response.status_code == 200
        assert response.json()['join'] is True and response.json()['join_probability'] == .8
        assert len(backend.calls) == 1
        assert client.get('/health').json()['participation']['healthy'] is True
    assert backend.closed


def test_disabled_gate_does_not_start_or_call_model(settings, embedder, index):
    backend = Backend()
    with TestClient(app_for(replace(settings, participation_enabled=False), embedder, index, backend)) as client:
        assert client.post('/v1/participation/decide', json=warmup_state()).status_code == 503
        assert client.get('/health').json()['participation'] == {'enabled': False}
    assert backend.calls == [] and not backend.closed


def test_busy_gate_returns_503_not_a_fake_quiet_decision(settings, embedder, index):
    settings = replace(settings, participation_enabled=True, participation_timeout=.01)
    with TestClient(app_for(settings, embedder, index, Backend(delay=1))) as client:
        assert client.post('/v1/participation/decide', json=warmup_state()).status_code == 503


def test_native_argmax_and_invalid_probabilities(settings):
    backend = ParticipationClassifier(settings)
    try:
        backend._model = SimpleNamespace(decide=lambda *a, **k: [{'probs': dict(zip(OPTIONS, (.2, .8)))}])
        assert backend._decide(warmup_state()) == ParticipationDecision('quiet', .2)
        backend._model = SimpleNamespace(decide=lambda *a, **k: [{'probs': dict(zip(OPTIONS, (float('nan'), .8)))}])
        with pytest.raises(ValueError, match='probabilities'):
            backend._decide(warmup_state())
    finally:
        backend._worker.shutdown(wait=True)


def test_native_pool_and_cpu_worker_are_separate_from_routing(settings):
    settings = replace(settings, participation_model='lidoll/decider-chat-0-1')  # Explicit legacy mode retains separate workers.
    pool = make_participation(replace(settings, decider_gpu_layers=-1))
    assert isinstance(pool, DeciderPool) and pool._result_factory is ParticipationDecision
    worker = make_participation(replace(settings, decider_gpu_layers=0))
    assert isinstance(worker, ParticipationClassifier)
    asyncio.run(worker.aclose())


def test_participation_pool_preserves_typed_result(settings):
    from test_decider_pool import ControlledWorker, card, ready

    async def check():
        async def discover():
            return [card(0)]
        pool = DeciderPool(replace(settings, decider_gpu_layers=-1), discover=discover,
                           worker_factory=ControlledWorker, result_factory=ParticipationDecision)
        await pool.start()
        try:
            pending = asyncio.create_task(pool.decide(warmup_state()))
            await ready(lambda: pool._workers[0].pending is not None)
            pool._workers[0].pending.set_result(('join', .8))
            assert await pending == ParticipationDecision('join', .8)
        finally:
            await pool.aclose()
    asyncio.run(check())


def test_git_configs_match_participation_and_reject_routing_schema():
    configs = repository_configs()
    config, schema = json.loads(configs['decider_config.json']), json.loads(configs['routing_policy.json'])
    validate_checkpoint(config, schema)
    schema['policy'] = 'game-chat-v3'
    with pytest.raises(ValueError, match='checkpoint'):
        validate_checkpoint(config, schema)


def test_merge_requires_explicit_task_and_full_run():
    from merge_classifier import validate_run
    from train_classifier import task_schema
    metadata = {**task_schema('participation', 'schema_first'), 'prompt_layout': 'schema_first'}
    validate_run(metadata, {'completed': True, 'smoke_test': False}, 'participation')
    with pytest.raises(ValueError):
        validate_run(metadata, {'completed': True, 'smoke_test': False}, 'routing')
    with pytest.raises(ValueError):
        validate_run(metadata, {'completed': True, 'smoke_test': True}, 'participation')
