"""Protect contextual labels, partition boundaries and the separate JOIN/QUIET task."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from build_participation_dataset import build, context
from participation_data import validate_rows
from train_classifier import load_dataset, load_development, report_rows, task_schema, encode_rows
from npc_rag.participation_policy import encode_state, build_participation_item
from npc_rag.decider_prompt import checkpoint_layout


def test_pilot_context_pairs_and_family_partitions():
    rows = build()
    splits = validate_rows(rows, ('train', 'validation', 'development', 'test'))
    assert {k: len(v) for k, v in splits.items()} == {'train': 160, 'validation': 32, 'development': 32, 'test': 32}
    assert len({r['family_id'] for r in rows}) == 64
    for group in splits.values():
        assert sum(r['label'] == 'join' for r in group) == len(group) // 2
    # The same words must sometimes require different decisions; message-only shortcuts fail.
    pairs = [r for r in rows if r['family_id'] == 'followup-travel' and r['state']['candidate']['kind'] == 'npc']
    assert pairs[0]['state']['message'] == pairs[1]['state']['message']
    assert pairs[0]['label'] != pairs[1]['label']
    assert pairs[0]['state']['recent_turns'] != pairs[1]['state']['recent_turns']
    train_names = {r['state']['candidate']['name'] for r in splits['train']}
    test_names = {r['state']['candidate']['name'] for r in splits['test']}
    assert not train_names & test_names
    for name in train_names:
        assert {r['label'] for r in splits['train'] if r['state']['candidate']['name'] == name} == {'join', 'quiet'}


def test_checked_in_pilot_matches_builder_and_trainer():
    splits = load_dataset(ROOT / 'data/participation/synthetic-pilot-v1.jsonl', task='participation')
    development = load_development(ROOT / 'data/participation/development-v1.jsonl', splits, task='participation')
    actual = sorted([r for group in splits.values() for r in group] + development, key=lambda r: r['id'])
    assert actual == sorted(build(), key=lambda r: r['id'])
    manifest = json.loads((ROOT / 'data/participation/manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['sha256_utf8_lf'].items():
        content = (ROOT / 'data/participation' / name).read_text(encoding='utf-8')
        assert hashlib.sha256(content.encode('utf-8')).hexdigest() == expected


@pytest.mark.parametrize('field', ('family_id', 'split_group'))
def test_cross_split_family_is_rejected(field):
    rows = build()
    train = next(r for r in rows if r['split'] == 'train')
    test = next(r for r in rows if r['split'] == 'test')
    test[field] = train[field]
    with pytest.raises(ValueError, match='cross data splits'):
        validate_rows(rows, ('train', 'validation', 'development', 'test'))


def test_same_message_in_new_context_cannot_leak_into_test():
    rows = build()
    train = next(r for r in rows if r['split'] == 'train')
    test = next(r for r in rows if r['split'] == 'test')
    test['state']['message'] = train['state']['message'].upper()
    with pytest.raises(ValueError, match='cross data splits'):
        validate_rows(rows, ('train', 'validation', 'development', 'test'))


def test_development_overlap_is_rejected(tmp_path):
    splits = load_dataset(ROOT / 'data/participation/synthetic-pilot-v1.jsonl', task='participation')
    rows = deepcopy(splits['train'][:4])
    for row in rows:
        row['split'] = 'development'
    path = tmp_path / 'dev.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate|cross data splits'):
        load_development(path, splits, task='participation')


@pytest.mark.parametrize('change', ('label', 'too_long', 'too_many_turns', 'string_flag', 'unknown_role'))
def test_bad_context_and_label_leakage_are_rejected(change):
    state = context('Could you explain that?')
    if change == 'label':
        state['label'] = 'join'
    elif change == 'too_long':
        state['message'] = 'x' * 501
    elif change == 'too_many_turns':
        state['recent_turns'] = [{'speaker': 'Mira', 'text': 'Hello'}] * 5
    elif change == 'string_flag':
        state['conversation_with_candidate'] = 'false'
    else:
        state['candidate']['kind'] = 'admin'
    with pytest.raises(ValueError):
        encode_state(state)


def test_interruptions_and_missed_joins_report_separately():
    report = report_rows([{'expected': 'quiet', 'predicted': 'join'},
                          {'expected': 'join', 'predicted': 'quiet'},
                          {'expected': 'join', 'predicted': 'join'}], task='participation')
    assert report['unwanted_interruptions'] == report['missed_joins'] == 1
    assert report['join_precision'] == .5
    assert not report['gate_passed']
    silent = report_rows([{'expected': 'quiet', 'predicted': 'quiet'},
                          {'expected': 'join', 'predicted': 'quiet'}], task='participation')
    assert silent['join_precision'] is None and not silent['gate_passed']


def test_participation_checkpoint_cannot_load_as_routing():
    schema = task_schema('participation', 'schema_first')
    assert schema['options'] == ['Join the conversation', 'Stay quiet']
    with pytest.raises(ValueError, match='disagrees'):
        checkpoint_layout({'routing_policy': schema['policy'], 'routing_prompt_order': 'schema_first'})


def test_encoder_uses_context_not_labels_and_obeys_permutations(monkeypatch):
    import types
    observed = []

    def fake_build(example, tokenizer, *, rng, **kwargs):
        order = [0, 1]
        rng.shuffle(order)
        observed.append((example.context, example.qs[0], kwargs))
        return {'ids': [1, 2, 3], 'slots': [2], 'golds': [order.index(example.qs[0].gold)], 'perms': [order]}

    monkeypatch.setitem(sys.modules, 'decider.prompt', types.SimpleNamespace(build=fake_build))
    tokenizer = types.SimpleNamespace(encode=lambda *a, **k: [1])
    row = build()[0]
    item = encode_rows([row], tokenizer, 768, layout='schema_first', task='participation', shuffle_options=True)[0]
    state, question, options = observed[0]
    assert json.loads(state) == row['state']
    assert row['id'] not in state and 'synthetic_draft' not in state and 'label' not in json.loads(state)
    assert item['perms'][0][item['golds'][0]] == 0
    assert options['layout'] == 'schema_first'
    assert question.options == ['Join the conversation', 'Stay quiet']


def test_prompt_does_not_silently_truncate(monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, 'decider.prompt', types.SimpleNamespace(build=lambda *a, **k: {'ids': [1] * 769, 'slots': [768]}))
    tokenizer = types.SimpleNamespace(encode=lambda *a, **k: [1])
    with pytest.raises(ValueError, match='Full participation prompt exceeds'):
        build_participation_item(context('Hello'), tokenizer, 768)
