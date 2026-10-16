"""Optional JOIN/QUIET native workers, separate from the GAME/CHAT model and its cache."""
import json
import math
from pathlib import Path
import time
from typing import NamedTuple

from .classifier_backend import DeciderClassifier
from .config import ROOT
from .decider_pool import DeciderPool, ProcessWorker
from .decider_devices import pinned_device
from .model_download import CONFIG_FILES, ensure_custom_decider
from .participation_policy import (OPTIONS, QUESTION, create_participation_decider, encode_state,
                                   policy_for_layout, validate_checkpoint)

MODEL = 'lidoll/decider-chat-0-1'
GGUF = 'decider-chat-0-1.gguf'
CONFIG_DIRECTORY = Path(__file__).with_name('participation_config')
RELEASE_MANIFEST = Path(__file__).with_name('participation_release.json')


class ParticipationDecision(NamedTuple):
    label: str
    join_probability: float


def repository_configs():
    configs = {name: (CONFIG_DIRECTORY / name).read_bytes() for name in CONFIG_FILES}
    config = json.loads(configs['decider_config.json'])
    validate_checkpoint(config, json.loads(configs['routing_policy.json']))
    temperature = float(config['temperature'])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Participation temperature must be finite and positive')
    for name, data in configs.items():
        if name.endswith('.json'):
            json.loads(data)
    return configs  # Small tunable configs follow Git; hosted weights and tokenizer stay checksum-pinned.


def ensure_participation(destination=None):
    return ensure_custom_decider(destination, release_manifest=RELEASE_MANIFEST, config_loader=repository_configs,
                                 gguf_file=GGUF, model_name='decider-chat-0-1')


def warmup_state():
    return {'candidate': {'name': 'Mira', 'kind': 'companion', 'role': 'traveler', 'interests': []},
            'speaker': 'Ari', 'message': 'Mira, how are you?', 'kind': 'chat', 'nearby_names': ['Mira'],
            'recent_turns': [], 'conversation_with_candidate': False, 'speaker_hired_candidate': False}


class ParticipationClassifier(DeciderClassifier):
    def _load(self):
        if self._model is not None:
            return
        source = self.settings.participation_model
        if source == MODEL:
            source = str(ensure_participation())
        folder = Path(source)
        if not folder.is_absolute():
            folder = ROOT / folder
        if not folder.is_dir():
            raise ValueError('PARTICIPATION_MODEL must be the managed alias or a local release directory')
        import llama_cpp
        from .native_logging import install_native_error_logging
        install_native_error_logging(llama_cpp)
        self._model_source, self._gguf_file = str(folder), GGUF
        self._model = create_participation_decider(folder, gguf_file=GGUF,
                        gguf_options={'n_gpu_layers': self.settings.decider_gpu_layers,
                                      'n_threads': self.settings.decider_threads, 'n_ctx': self.settings.decider_context,
                                      'n_seq_max': 1, 'verbose': True})

    async def start(self):
        await super().start()
        await self._native(lambda: self._decide(warmup_state()))  # CPU workers also prove decode succeeds before readiness.

    def _decide(self, state):
        rows = self._model.decide(encode_state(state), [{'question': QUESTION, 'options': list(OPTIONS)}], max_ctx_tokens=768)
        if len(rows) != 1 or set(rows[0]['probs']) != set(OPTIONS):
            raise ValueError('Invalid participation result')
        values = [float(rows[0]['probs'][option]) for option in OPTIONS]
        if (any(not math.isfinite(p) or not 0 <= p <= 1 for p in values)
                or not math.isclose(sum(values), 1, abs_tol=1e-4)):
            raise ValueError('Invalid participation probabilities')
        label = 'join' if values[0] >= values[1] else 'quiet'
        return ParticipationDecision(label, values[0])  # Same argmax rule as training/evaluation; scores are uncalibrated.

    def view(self):
        return {'backend': 'decider', 'task': 'participation', 'decision_policy': policy_for_layout('schema_first'),
                'model': self._model_source, 'gguf_file': GGUF, 'release': MODEL,
                'prompt_layout': 'schema_first', 'prefix_cache': False, 'capacity': 1,
                'busy': self._busy, 'failures': self._failures, 'gpu_layers': self.settings.decider_gpu_layers}

    async def model_identity(self):
        return ('decider', self._model_source, GGUF, policy_for_layout('schema_first')) if await self.health() else None


_worker_backend = None


def _start(settings, device):
    import os
    global _worker_backend
    _worker_backend = ParticipationClassifier(settings)
    with pinned_device(device['name']) as actual:
        _worker_backend._load()
    began = time.perf_counter()
    _worker_backend._decide(warmup_state())
    metadata = _worker_backend.view()
    metadata['startup_inference_ms'] = round((time.perf_counter() - began) * 1000)
    return {'metadata': metadata, 'device': actual, 'pid': os.getpid()}


def _decide(state):
    try:
        return tuple(_worker_backend._decide(state))
    except Exception as exc:
        raise RuntimeError(f'Participation inference failed ({type(exc).__name__})') from None


def _stop():
    global _worker_backend
    if _worker_backend is not None:
        _worker_backend._unload()
        _worker_backend._worker.shutdown(wait=True)
        _worker_backend = None


def _worker(device):
    return ProcessWorker(device, operations=(_start, _decide, _stop))


def make_participation(settings, *, classifier=None):
    from .multitask import RELEASES, TaskView
    if settings.participation_model == 'shared' or settings.participation_model in RELEASES:
        if settings.participation_model in RELEASES and settings.participation_model != settings.decider_model:
            raise ValueError('Shared participation alias must match DECIDER_MODEL')
        if not isinstance(classifier, TaskView) or classifier.task != 'routing':
            raise ValueError('Shared participation requires CLASSIFIER_BACKEND=decider and a multitask DECIDER_MODEL')
        return TaskView(classifier.backend, 'participation')  # Share admission, model memory and lifecycle with routing.
    if settings.decider_gpu_layers != 0:
        return DeciderPool(settings, worker_factory=_worker, result_factory=ParticipationDecision)
    return ParticipationClassifier(settings)  # GPU mode reuses the existing one-worker-per-card admission/lifetime logic.
