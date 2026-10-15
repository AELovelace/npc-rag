"""Protect task balance, production prompts, and per-task evaluation in the combined run."""
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from multitask_data import load_multitask, loss_weights, select_rows, augment_multitask, checkpoint_config
from train_classifier import encode_rows, report_rows


def arguments(**changes):
    return SimpleNamespace(**dict(dict(dataset=None, dev_dataset=None, participation_dataset=None,
                                      participation_dev_dataset=None), **changes))


def test_both_datasets_keep_holdouts_and_have_equal_loss_influence():
    splits, sources = load_multitask(arguments())
    assert {split: len(rows) for split, rows in splits.items()} == dict(train=960, validation=132, test=132, development=232)
    weights = loss_weights(splits['train'])
    assert weights == {'routing': .6, 'participation': 3}
    totals = {task: sum(weights[task] for row in splits['train'] if row['task'] == task) for task in weights}
    assert totals['routing'] == pytest.approx(totals['participation'])
    assert sources['routing']['prompt_layout'] == 'state_first'
    assert sources['participation']['prompt_layout'] == 'schema_first'
    smoke = select_rows(splits, True)
    for rows in smoke.values():
        assert Counter(row['task'] for row in rows) == {'routing': 8, 'participation': 8}
    augmented, stats = augment_multitask(splits['train'], 417)
    assert len(augmented) == 960
    assert [r for r in augmented if r['task'] == 'participation'] == [r for r in splits['train'] if r['task'] == 'participation']
    assert stats['loss_weights'] == weights


def test_cross_task_training_overlap_with_heldout_is_rejected(tmp_path):
    path = ROOT / 'data/participation/synthetic-pilot-v1.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    fixed = json.loads((ROOT / 'data/eval_messages.json').read_text(encoding='utf-8'))
    next(row for row in rows if row['split'] == 'train')['state']['message'] = fixed['game'][0]
    changed = tmp_path / 'participation.jsonl'
    changed.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    with pytest.raises(ValueError, match='held-out'):
        load_multitask(arguments(participation_dataset=changed))


def test_one_successful_task_cannot_hide_other_task_failure():
    rows = [{'task': 'routing', 'expected': label, 'predicted': label} for label in ('game', 'general') for _ in range(10)]
    rows += [{'task': 'participation', 'expected': 'join', 'predicted': 'quiet'},
             {'task': 'participation', 'expected': 'quiet', 'predicted': 'quiet'}]
    result = report_rows(rows, task='multitask')
    assert result['accuracy'] == .75  # Each task contributes half, regardless of its test-set size.
    assert not result['gate_passed']
    assert len(result['by_class']) == 4
    assert result['by_task']['participation']['missed_joins'] == 1


def test_mixed_encoder_keeps_task_questions_and_gold_remapping(monkeypatch):
    observed = []

    def fake_build(example, tokenizer, *, rng, **kwargs):
        order = [0, 1]
        rng.shuffle(order)
        observed.append((example.qs[0].text, kwargs['layout'], kwargs['max_ctx_tokens']))
        return {'ids': [1, 2, 3], 'slots': [2], 'golds': [order.index(example.qs[0].gold)], 'perms': [order]}

    monkeypatch.setitem(sys.modules, 'decider.prompt', SimpleNamespace(build=fake_build))
    splits, _ = load_multitask(arguments())
    rows = [next(row for row in splits['train'] if row['task'] == task) for task in ('routing', 'participation')]
    encoded = encode_rows(rows, SimpleNamespace(encode=lambda *a, **k: [1]), 768, task='multitask', shuffle_options=True)
    assert [(layout, budget) for _, layout, budget in observed] == [('state_first', 384), ('schema_first', 768)]
    assert observed[0][0] != observed[1][0]
    assert [item['task'] for item in encoded] == ['routing', 'participation']
    assert all(item['perms'][0][item['golds'][0]] == 0 for item in encoded)


def test_experimental_checkpoint_cannot_silently_load_as_routing():
    from npc_rag.decider_prompt import checkpoint_layout
    with pytest.raises(ValueError, match='disagrees'):
        checkpoint_layout(checkpoint_config())


def test_merge_requires_explicit_multitask_and_both_exact_schemas():
    from copy import deepcopy
    from merge_classifier import validate_run
    from multitask_data import POLICY
    _, sources = load_multitask(arguments())
    metadata = {'task': 'multitask', 'policy': POLICY, 'prompt_layout': 'per_task', 'tasks': sources}
    completion = {'completed': True, 'smoke_test': False}
    assert validate_run(metadata, completion, 'multitask')['policy'] == POLICY
    with pytest.raises(ValueError):
        validate_run(metadata, completion, 'routing')
    with pytest.raises(ValueError, match='completed full'):
        validate_run(metadata, {**completion, 'smoke_test': True}, 'multitask')
    damaged = deepcopy(metadata)
    damaged['tasks']['participation']['question'] = 'Wrong question'
    with pytest.raises(ValueError, match='schema'):
        validate_run(damaged, completion, 'multitask')


def test_specialist_comparison_flags_regression_and_refuses_mismatched_sources(tmp_path):
    from compare_multitask_runs import compare
    from multitask_data import POLICY
    _, sources = load_multitask(arguments())
    metadata = {'task': 'multitask', 'policy': POLICY, 'tasks': sources, 'model': 'fixture', 'revision': 'fixed'}
    run = tmp_path / 'combined'
    run.mkdir()

    def write(folder, name, value):
        (folder / name).write_text(json.dumps(value), encoding='utf-8')

    write(run, 'run.json', metadata)
    write(run, 'completion.json', {'completed': True, 'smoke_test': False})
    rows, specialists = [], {}
    for task, labels in (('routing', ('game', 'general')), ('participation', ('join', 'quiet'))):
        specialist = tmp_path / task
        specialist.mkdir()
        specialists[task] = specialist
        specialist_rows = [{'id': label, 'task': task, 'expected': label, 'predicted': label} for label in labels]
        write(specialist, 'run.json', {**sources[task], 'model': 'fixture', 'revision': 'fixed'})
        write(specialist, 'completion.json', {'completed': True, 'smoke_test': False})
        write(specialist, 'final-test.json', {**report_rows(specialist_rows, task=task), 'rows': specialist_rows})
        rows.extend(specialist_rows)
    write(run, 'final-test.json', {**report_rows(rows, task='multitask'), 'rows': rows})
    assert compare(run, specialists)['both_tasks_without_test_regression']
    rows[-2]['predicted'] = 'quiet'
    write(run, 'final-test.json', {**report_rows(rows, task='multitask'), 'rows': rows})
    result = compare(run, specialists)
    assert not result['both_tasks_without_test_regression']
    assert result['tasks']['participation']['recall_deltas']['join'] == -1
    metadata['tasks']['routing']['dataset_sha256'] = 'changed'
    write(run, 'run.json', metadata)
    with pytest.raises(ValueError, match='dataset_sha256'):
        compare(run, specialists)


def test_native_evaluator_preserves_task_order_and_rejects_invalid_probabilities(monkeypatch):
    import evaluate_multitask as evaluator
    seen = []

    def fake_encode(rows, tokenizer, budget, *, layout, task):
        seen.append((task, layout, budget))
        return [{'fake': True}]

    monkeypatch.setattr(evaluator, 'encode_rows', fake_encode)
    engine = SimpleNamespace(m=SimpleNamespace(tok=None))

    class Probabilities:
        def __getitem__(self, key):
            return self

        def tolist(self):
            return [.75, .25]

    engine.score_items = lambda *a, **k: [Probabilities()]
    rows = [{'id': 'one', 'task': 'participation', 'label': 'join'}, {'id': 'two', 'task': 'routing', 'label': 'game'}]
    results = evaluator.evaluate_engine(engine, rows)
    assert [row['predicted'] for row in results] == ['join', 'game']
    assert seen == [('participation', 'schema_first', 768), ('routing', 'state_first', 384)]
    monkeypatch.setattr(Probabilities, 'tolist', lambda self: [float('nan'), .5])
    with pytest.raises(ValueError, match='probabilities'):
        evaluator.evaluate_engine(engine, rows)
