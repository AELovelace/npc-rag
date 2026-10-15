"""Serve routing and participation through one cancellation-safe native pool."""
import json
import math
import os
from pathlib import Path
import time

from .classifier import ModelDecision
from .classifier_backend import DeciderClassifier
from .config import ROOT
from .decider_devices import pinned_device
from .decider_policy import DECIDER_OPTIONS, DECIDER_QUESTION, DECIDER_POLICY
from .decider_pool import DeciderPool, ProcessWorker
from .decider_prompt import build_decision_item
from .model_download import CONFIG_FILES, ensure_custom_decider
from .participation_policy import (OPTIONS, QUESTION, build_participation_item,
                                   policy_for_layout)

MODEL = 'lidoll/decider-multitask-v2'
GGUF = 'decider-multitask-v2.gguf'
POLICY = 'routing-participation-multitask-v1'
LAYOUTS = {'routing': 'state_first', 'participation': 'schema_first'}
BUDGETS = {'routing': 384, 'participation': 768}
CONFIG_DIRECTORY = Path(__file__).with_name('multitask_config')
RELEASE_MANIFEST = Path(__file__).with_name('multitask_release.json')


def validate_checkpoint(config, schema):
    if (config.get('routing_policy') != POLICY or config.get('task') != 'multitask'
            or config.get('layout') != 'plain' or config.get('task_prompt_layouts') != LAYOUTS
            or config.get('temperature') != 1.0 or config.get('temperature_by_type')
            or schema.get('policy') != POLICY or schema.get('task') != 'multitask'):
        raise ValueError('Unexpected multitask checkpoint configuration')
    expected = {'routing': (DECIDER_POLICY, DECIDER_QUESTION, DECIDER_OPTIONS),
                'participation': (policy_for_layout('schema_first'), QUESTION, OPTIONS)}
    if set(schema.get('tasks', {})) != set(expected):
        raise ValueError('Multitask checkpoint must contain both tasks')
    for task, (policy, question, options) in expected.items():
        entry = schema['tasks'][task]
        if (entry.get('policy') != policy or entry.get('question') != question
                or entry.get('options') != list(options) or entry.get('prompt_layout') != LAYOUTS[task]
                or entry.get('max_tokens') != BUDGETS[task]):
            raise ValueError('Multitask checkpoint task schema does not match serving')
    # Validate before allocation so a specialist or altered question cannot load accidentally.


def repository_configs():
    configs = {name: (CONFIG_DIRECTORY / name).read_bytes() for name in CONFIG_FILES}
    for name, content in configs.items():
        if name.endswith('.json'):
            json.loads(content)
    validate_checkpoint(json.loads(configs['decider_config.json']), json.loads(configs['routing_policy.json']))
    return configs  # Git owns the small configs; hosted model/tokenizer assets remain hash-pinned.


def ensure_multitask(destination=None):
    return ensure_custom_decider(destination, release_manifest=RELEASE_MANIFEST, config_loader=repository_configs,
                                 gguf_file=GGUF, model_name='decider-multitask-v2')


class MultitaskClassifier(DeciderClassifier):
    """One engine scores both trained prompt layouts using the offline evaluator's A/B path."""

    def _load(self):
        if self._model is not None:
            return
        source = self.settings.decider_model
        folder = ensure_multitask() if source == MODEL else Path(source)
        if not folder.is_absolute():
            folder = ROOT / folder
        validate_checkpoint(json.loads((folder / 'decider_config.json').read_text(encoding='utf-8')),
                            json.loads((folder / 'routing_policy.json').read_text(encoding='utf-8')))
        import llama_cpp
        from decider.engine_gguf import GGUFEngine
        from .native_logging import install_native_error_logging
        install_native_error_logging(llama_cpp)
        self._model_source, self._gguf_file = str(folder), GGUF
        self._model = GGUFEngine(str(folder / GGUF), str(folder),
                                n_gpu_layers=self.settings.decider_gpu_layers,
                                n_threads=self.settings.decider_threads, n_ctx=self.settings.decider_context,
                                n_seq_max=1, verbose=True)  # Both tasks reuse this exact allocation.

    def _decide(self, request):
        task, payload = request
        if task == 'routing':
            item = build_decision_item(payload, self._model.m.tok, BUDGETS[task], layout=LAYOUTS[task])
            labels = ('game', 'general')
        elif task == 'participation':
            item = build_participation_item(payload, self._model.m.tok, BUDGETS[task], layout=LAYOUTS[task])
            labels = ('join', 'quiet')
        else:
            raise ValueError('Unknown multitask request')
        probs = self._model.score_items([item], temperature=1.0)[0][0][:2].tolist()
        if (len(probs) != 2 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs)
                or not math.isclose(sum(probs), 1, abs_tol=1e-4)):
            raise ValueError('Invalid multitask probabilities')
        return labels[0 if probs[0] >= probs[1] else 1], probs[0]  # Preserve the first-option probability for each task.

    def _warmup(self):
        from .participation import warmup_state
        self._decide(('routing', 'A quiet afternoon sounds pleasant.'))
        self._decide(('participation', warmup_state()))  # Readiness requires both prompt paths to decode.

    async def start(self):
        await super().start()
        await self._native(self._warmup)

    def view(self):
        return {**super().view(), 'release': MODEL, 'decision_policy': POLICY,
                'prompt_layout': 'per_task', 'tasks': list(LAYOUTS), 'shared_pool': True,
                'calibration': 'uncalibrated'}

    async def model_identity(self):
        return ('decider', self._model_source, GGUF, POLICY) if await self.health() else None

    def _unload(self):
        engine, self._model = self._model, None
        if engine is not None:
            engine.__del__()
            engine.L = None  # Upstream destructor leaves pointers set; prevent a second native free.


_worker_backend = None


def _start(settings, device):
    global _worker_backend
    _worker_backend = MultitaskClassifier(settings)
    with pinned_device(device['name']) as actual:
        _worker_backend._load()
    began = time.perf_counter()
    _worker_backend._warmup()
    metadata = _worker_backend.view()
    metadata['startup_inference_ms'] = round((time.perf_counter() - began) * 1000)
    return {'metadata': metadata, 'device': actual, 'pid': os.getpid()}


def _decide(request):
    try:
        return _worker_backend._decide(request)
    except Exception as exc:
        raise RuntimeError(f'Multitask inference failed ({type(exc).__name__})') from None


def _stop():
    global _worker_backend
    if _worker_backend is not None:
        _worker_backend._unload()
        _worker_backend._worker.shutdown(wait=True)
        _worker_backend = None


def _worker(device):
    return ProcessWorker(device, operations=(_start, _decide, _stop))


def _result(label, probability):
    return label, probability  # A module-level factory is safe for spawned workers and either task's labels.


class TaskView:
    """A typed task interface; routing owns startup/shutdown of the shared backend."""

    def __init__(self, backend, task):
        self.backend, self.task = backend, task

    async def start(self):
        if self.task == 'routing':
            await self.backend.start()
        elif not await self.backend.health():
            raise RuntimeError('Start the shared routing backend before participation')

    async def decide(self, payload):
        from .participation import ParticipationDecision
        label, probability = await self.backend.decide((self.task, payload))
        factory = ModelDecision if self.task == 'routing' else ParticipationDecision
        return factory(label, probability)

    async def health(self):
        return await self.backend.health()

    async def model_identity(self):
        identity = await self.backend.model_identity()
        return (*identity, self.task) if identity else None  # Keep task identity explicit for caches.

    def view(self):
        return {**self.backend.view(), 'task': self.task, 'prompt_layout': LAYOUTS[self.task],
                'shared_pool': True}  # Capacity/busy counts describe the same pool in both health sections.

    async def aclose(self):
        if self.task == 'routing':
            await self.backend.aclose()  # The participation view never frees another task's running engine.


def make_multitask(settings):
    backend = (DeciderPool(settings, worker_factory=_worker, result_factory=_result)
               if settings.decider_gpu_layers != 0 else MultitaskClassifier(settings))
    return TaskView(backend, 'routing')


def is_multitask_source(source):
    if source == MODEL:
        return True
    folder = Path(source)
    config = (folder if folder.is_absolute() else ROOT / folder) / 'decider_config.json'
    return config.is_file() and json.loads(config.read_text(encoding='utf-8')).get('routing_policy') == POLICY
