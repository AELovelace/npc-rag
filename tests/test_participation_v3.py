"""Preserve prior regressions and require contextual, balanced opening-conversation pairs."""
from collections import Counter, defaultdict
from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(os.environ.get('NPC_V3_REPO', Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from build_participation_v3 import build, validate, read_rows
from participation_scenarios_v3 import NAMES
from participation_data import validate_rows
from multitask_data import load_multitask


def test_v3_preserves_v2_and_balances_every_new_split():
    rows = build(ROOT)
    validate(ROOT, rows)
    splits = validate_rows(rows, ('train', 'validation', 'development', 'test'))
    assert {split: len(group) for split, group in splits.items()} == dict(train=1120, validation=136, development=136, test=176)
    for split, group in splits.items():
        assert Counter(row['label'] for row in group) == {'join': len(group) // 2, 'quiet': len(group) // 2}
        new = [row for row in group if row['id'].startswith('participation-v3-')]
        assert len({row['family_id'] for row in new}) == (40 if split == 'train' else 8)
        for other, names in NAMES.items():
            if other != split:
                assert set(NAMES[split]).isdisjoint(names)


def test_hiring_contrast_and_explicit_boundaries_are_present_in_every_family():
    families = defaultdict(dict)
    for row in build(ROOT):
        if row['id'].startswith('participation-v3-'):
            families[row['family_id']][row['scenario']] = row
    for variants in families.values():
        assert len(variants) == 8
        positive, negative = variants['hired_opening'], variants['unaddressed_bystander']
        assert positive['label'] == 'join' and negative['label'] == 'quiet'
        comparable = deepcopy(negative['state'])
        comparable['speaker_hired_candidate'] = True
        assert comparable == positive['state']  # Only the hiring flag changes in the core contrast pair.
        assert not positive['state']['recent_turns'] and not positive['state']['conversation_with_candidate']
        for scenario in ('other_addressee', 'quoted_address', 'requested_silence'):
            assert variants[scenario]['label'] == 'quiet'
            assert variants[scenario]['state']['speaker_hired_candidate'] is True
        assert variants['nonhired_ongoing']['label'] == 'join'
        assert variants['explicit_address']['label'] == 'join'


def test_related_variants_cannot_leak_into_a_different_split():
    rows = build(ROOT)
    next(row for row in rows if row['id'].startswith('participation-v3-train') or
         (row['id'].startswith('participation-v3-') and row['split'] == 'train'))['split'] = 'test'
    with pytest.raises(ValueError, match='cross data splits'):
        validate_rows(rows, ('train', 'validation', 'development', 'test'))


def test_generated_data_matches_builder_and_cross_task_checks(tmp_path):
    rows = build(ROOT)
    import json
    for name, selected in (('data.jsonl', [row for row in rows if row['split'] != 'development']),
                           ('dev.jsonl', [row for row in rows if row['split'] == 'development'])):
        (tmp_path / name).write_text(''.join(json.dumps(row) + '\n' for row in selected), encoding='utf-8')
    splits, _ = load_multitask(SimpleNamespace(dataset=None, dev_dataset=None,
                      participation_dataset=tmp_path / 'data.jsonl', participation_dev_dataset=tmp_path / 'dev.jsonl'))
    assert len(splits['train']) == 1920
