"""Expand the participation pilot while retaining every v1 row and its original split."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_participation_dataset import build as build_v1
from participation_data import validate_rows
import participation_scenarios_v2 as scenes

NAMES = {'train': ('Nima', 'Oren', 'Pella', 'Brin', 'Fenn', 'Zuri'),
         'validation': ('Aven', 'Eris', 'Moss', 'Tova'),
         'development': ('Bex', 'Cleo', 'Dune', 'Elin'),
         'test': ('Enna', 'Faro', 'Galen', 'Hest')}


def state(message):
    return {'candidate': {'name': 'Nima', 'kind': 'companion', 'role': 'traveler', 'interests': ['local walks']},
            'speaker': 'Visitor', 'message': message, 'kind': 'chat', 'nearby_names': ['Nima', 'Oren'],
            'recent_turns': [], 'conversation_with_candidate': False, 'speaker_hired_candidate': False}


def pairs():
    output = []
    for index, (role, interest, message) in enumerate(scenes.EXPERTISE):
        yes, no = state(message), state(message)
        yes['candidate'].update(role=role, interests=[interest])
        unrelated = scenes.EXPERTISE[(index + 7) % len(scenes.EXPERTISE)]
        no['candidate'].update(role=unrelated[0], interests=[unrelated[1]])
        output.append(('expertise', index, yes, no))
    for category, entries in (('followup', scenes.FOLLOWUPS), ('handoff', scenes.HANDOFFS)):
        for index, (previous, message) in enumerate(entries):
            yes, no = state(message), state(message)
            yes['recent_turns'] = [{'speaker': 'Nima', 'text': previous}]
            no['recent_turns'] = [{'speaker': 'Oren', 'text': previous}]
            if category == 'handoff':
                # Both histories begin with the candidate; the latest exchange changes ownership.
                opening = [{'speaker': 'Visitor', 'text': 'Nima, thank you for showing me around.'},
                           {'speaker': 'Nima', 'text': 'You are welcome. Let us hear how the others are doing.'}]
                yes['recent_turns'] = opening + yes['recent_turns']
                no['recent_turns'] = opening + no['recent_turns']
            yes['conversation_with_candidate'] = no['conversation_with_candidate'] = bool(index % 2)
            output.append((category, index, yes, no))
    for category, entries in (('farewell', scenes.FAREWELLS), ('addressed', scenes.ADDRESSED),
                              ('invitation', scenes.INVITATIONS), ('emote', scenes.EMOTES)):
        for index, message in enumerate(entries):
            yes, no = state(message), state(message)
            no['candidate']['name'] = 'Oren'
            if category == 'emote':
                yes['kind'] = no['kind'] = 'emote'
            output.append((category, index, yes, no))
    for index, message in enumerate(scenes.OOC):
        yes, no = state(message), state(message)
        no['kind'] = 'ooc'
        yes['conversation_with_candidate'] = no['conversation_with_candidate'] = bool(index % 2)
        output.append(('ooc', index, yes, no))
    for category, entries in (('quotation', scenes.QUOTATIONS), ('boundary', scenes.BOUNDARIES)):
        for index, (positive, negative) in enumerate(entries):
            yes, no = state(positive), state(negative)
            yes['conversation_with_candidate'] = no['conversation_with_candidate'] = bool(index % 2)
            output.append((category, index, yes, no))
    return output  # Four rows per family represent two context decisions, each in two candidate roles.


def build():
    authored = pairs()
    categories = Counter(category for category, *_ in authored)
    if len(categories) != 10 or set(categories.values()) != {20}:
        raise ValueError('Expected ten categories with twenty authored scenario families each')
    assignments = {}
    rng = random.Random(20261015)
    for category in sorted(categories):
        order = list(range(20))
        rng.shuffle(order)
        for index, split in zip(order, ['train'] * 16 + ['validation', 'development'] + ['test'] * 2):
            assignments[category, index] = split
    rows = deepcopy(build_v1())  # Never move a known held-out family into training or rewrite its label.
    for category, index, yes, no in authored:
        split = assignments[category, index]
        family = f'v2-{category}-{index:02d}'
        names = NAMES[split]
        replacements = {'Nima': names[index % len(names)], 'Oren': names[(index + 1) % len(names)]}
        for kind in ('companion', 'npc'):
            for label, original in (('join', yes), ('quiet', no)):
                current = deepcopy(original)
                current['candidate']['kind'] = kind
                current['speaker_hired_candidate'] = kind == 'companion' and index % 2 == 0
                current = json.loads(re.sub(r'\b(Nima|Oren)\b', lambda match: replacements[match[0]], json.dumps(current)))
                rows.append({'id': f'participation-v2-{category}-{index:02d}-{kind}-{label}',
                             'family_id': family, 'split_group': family, 'split': split,
                             'label': label, 'state': current, 'origin': 'assistant_authored_synthetic',
                             'review_status': 'synthetic_draft', 'scenario': category})
    validate_rows(rows, ('train', 'validation', 'development', 'test'))
    return rows


def main():
    rows = build()
    target = ROOT / 'data/participation'
    hashes = {}
    for name, selected in (('synthetic-expanded-v2.jsonl', [r for r in rows if r['split'] != 'development']),
                           ('development-v2.jsonl', [r for r in rows if r['split'] == 'development'])):
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected)
        (target / name).write_text(content, encoding='utf-8', newline='\n')
        hashes[name] = hashlib.sha256(content.encode('utf-8')).hexdigest()
    manifest = {'task': 'participation', 'total_rows': len(rows), 'families': len({r['family_id'] for r in rows}),
                'splits': dict(Counter(r['split'] for r in rows)), 'sha256_utf8_lf': hashes,
                'new_rows_by_split': dict(Counter(r['split'] for r in rows if r['id'].startswith('participation-v2-'))),
                'new_families_by_category': dict(Counter(category for category, *_ in pairs())),
                'split_seed': 20261015, 'preserves_v1_rows_and_splits': True,
                'review_status': 'synthetic_draft', 'real_exported_decisions': 0, 'deployment_ready': False,
                'limitations': ['Paired contexts and NPC/companion variants are not independent conversations.',
                               'The v1 held-out patterns informed category coverage; v1 scores are regression diagnostics.',
                               'New synthetic holdouts share scenario categories with training and are not real-player validation.']}
    (target / 'manifest-v2.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
