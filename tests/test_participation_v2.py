"""Verify preserved holdouts and contextual diversity before the expanded participation run."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from build_participation_v2 import build, pairs, NAMES
from build_participation_dataset import build as build_v1
from participation_data import validate_rows
from multitask_data import load_multitask, loss_weights


def test_v2_preserves_every_original_row_and_partition():
    rows = build()
    original = {row['id']: row for row in build_v1()}
    assert {row['id']: row for row in rows if row['id'] in original} == original
    splits = validate_rows(rows, ('train', 'validation', 'development', 'test'))
    assert {split: len(group) for split, group in splits.items()} == dict(train=800, validation=72, development=72, test=112)
    assert len({row['family_id'] for row in rows}) == 264
    for group in splits.values():
        assert Counter(row['label'] for row in group)['join'] == len(group) // 2
    new = [row for row in rows if row['id'] not in original]
    assert len(new) == 800 and len({row['family_id'] for row in new}) == 200
    for split, names in NAMES.items():
        assert all(set(names).isdisjoint(other) for other_split, other in NAMES.items() if other_split != split)


def test_new_categories_all_have_unseen_heldout_families():
    families = defaultdict(lambda: defaultdict(set))
    for row in build():
        if row['id'].startswith('participation-v2-'):
            families[row['scenario']][row['split']].add(row['family_id'])
    assert len(families) == 10
    for category in families.values():
        assert {split: len(group) for split, group in category.items()} == dict(train=16, validation=1, development=1, test=2)


def test_context_pairs_cannot_be_solved_from_message_alone():
    for category, _, yes, no in pairs():
        if category not in ('quotation', 'boundary'):
            assert yes['message'] == no['message']
        if category == 'ooc':
            assert yes['kind'] == 'chat' and no['kind'] == 'ooc'
            comparable = deepcopy(no)
            comparable['kind'] = 'chat'
            assert comparable == yes
        if category == 'handoff':
            assert yes['recent_turns'][:-1] == no['recent_turns'][:-1]
            assert yes['recent_turns'][-1]['speaker'] != no['recent_turns'][-1]['speaker']


def test_checked_in_data_matches_reproducible_builder_and_routing_has_no_overlap():
    args = SimpleNamespace(dataset=None, dev_dataset=None,
                           participation_dataset=ROOT / 'data/participation/synthetic-expanded-v2.jsonl',
                           participation_dev_dataset=ROOT / 'data/participation/development-v2.jsonl')
    splits, sources = load_multitask(args)
    assert loss_weights(splits['train']) == {'routing': 1, 'participation': 1}
    checked_in = [json.loads(line) for path in (args.participation_dataset, args.participation_dev_dataset)
                  for line in path.read_text(encoding='utf-8').splitlines()]
    assert sorted(checked_in, key=lambda row: row['id']) == sorted(build(), key=lambda row: row['id'])
    assert sources['participation']['split_counts']['train'] == 800


def test_context_variants_cannot_move_across_partitions():
    rows = build()
    row = next(row for row in rows if row['id'].startswith('participation-v2-') and row['split'] == 'train')
    row['split'] = 'test'
    with pytest.raises(ValueError, match='cross data splits'):
        validate_rows(rows, ('train', 'validation', 'development', 'test'))


def test_expansion_comparison_requires_original_rows_and_source_hashes(tmp_path):
    from compare_multitask_runs import validate_participation_expansion
    folder = ROOT / 'data/participation'
    previous = {'dataset': str(folder / 'synthetic-pilot-v1.jsonl')}
    current = {'dataset': str(folder / 'synthetic-expanded-v2.jsonl'), 'development': str(folder / 'development-v2.jsonl')}
    for metadata, dataset, development in ((previous, Path(previous['dataset']), folder / 'development-v1.jsonl'),
                                           (current, Path(current['dataset']), Path(current['development']))):
        metadata['dataset_sha256'] = hashlib.sha256(dataset.read_bytes()).hexdigest()
        metadata['development_sha256'] = hashlib.sha256(development.read_bytes()).hexdigest()
    validate_participation_expansion(current, previous)
    damaged = [json.loads(line) for line in Path(current['dataset']).read_text(encoding='utf-8').splitlines()]
    old_test = next(row for row in damaged if row['id'].startswith('participation-v1-') and row['split'] == 'test')
    old_test['split'] = 'train'
    path = tmp_path / 'moved-test.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in damaged), encoding='utf-8')
    changed = {**current, 'dataset': str(path), 'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match='preserve all original'):
        validate_participation_expansion(changed, previous)
    with pytest.raises(ValueError, match='changed since training'):
        validate_participation_expansion({**current, 'dataset_sha256': 'wrong'}, previous)


def test_expanded_comparison_scores_only_the_preserved_subset(tmp_path):
    from compare_multitask_runs import compare
    from multitask_data import POLICY, LAYOUTS
    from train_classifier import task_schema, load_dataset, report_rows
    args = SimpleNamespace(dataset=None, dev_dataset=None,
                           participation_dataset=ROOT / 'data/participation/synthetic-expanded-v2.jsonl',
                           participation_dev_dataset=ROOT / 'data/participation/development-v2.jsonl')
    splits, sources = load_multitask(args)
    combined = tmp_path / 'combined'
    combined.mkdir()

    def write(folder, name, value):
        (folder / name).write_text(json.dumps(value), encoding='utf-8')

    base = {'model': 'fixture', 'revision': 'fixed'}
    completion = {'completed': True, 'smoke_test': False}
    write(combined, 'run.json', {**base, 'task': 'multitask', 'policy': POLICY, 'tasks': sources})
    write(combined, 'completion.json', completion)
    results = [{'id': row['id'], 'task': row['task'], 'expected': row['label'], 'predicted': row['label']} for row in splits['test']]
    for row in results:
        if row['id'].startswith('participation-v2-'):
            row['predicted'] = 'quiet' if row['expected'] == 'join' else 'join'
    write(combined, 'final-test.json', {**report_rows(results, task='multitask'), 'rows': results})
    specialists = {}
    for task in ('routing', 'participation'):
        folder = tmp_path / task
        folder.mkdir()
        specialists[task] = folder
        source = sources[task]
        if task == 'participation':
            dataset = ROOT / 'data/participation/synthetic-pilot-v1.jsonl'
            development = ROOT / 'data/participation/development-v1.jsonl'
            source = {**task_schema(task, LAYOUTS[task]), 'prompt_layout': LAYOUTS[task], 'dataset': str(dataset),
                      'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                      'development_sha256': hashlib.sha256(development.read_bytes()).hexdigest()}
        examples = load_dataset(Path(source['dataset']), task=task)['test']
        old_rows = [{'id': row['id'], 'expected': row['label'], 'predicted': row['label']} for row in examples]
        write(folder, 'run.json', {**base, **source})
        write(folder, 'completion.json', completion)
        write(folder, 'final-test.json', {**report_rows(old_rows, task=task), 'rows': old_rows})
    with pytest.raises(ValueError, match='dataset_sha256'):
        compare(combined, specialists)
    report = compare(combined, specialists, allow_expanded_participation=True)['tasks']['participation']
    assert report['total'] == 32 and report['multitask_accuracy'] == 1
    assert report['comparison_partition'] == 'preserved v1 test subset'
    assert report['expanded_test_summary']['total'] == 112
    assert report['expanded_test_summary']['accuracy'] == pytest.approx(32 / 112)
