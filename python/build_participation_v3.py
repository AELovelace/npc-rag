"""Preserve v2 and add balanced context pairs for hired companions opening a conversation."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from participation_scenarios_v3 import SCENARIOS, NAMES, PROFILES


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def build(repo):
    source = repo / 'data/participation'
    rows = read_rows(source / 'synthetic-expanded-v2.jsonl') + read_rows(source / 'development-v2.jsonl')
    for split, messages in SCENARIOS.items():
        names = NAMES[split]
        for index, message in enumerate(messages):
            name, other = names[index % len(names)], names[(index + 1) % len(names)]
            role, interests = PROFILES[index % len(PROFILES)]
            base = {'candidate': {'name': name, 'kind': 'companion', 'role': role, 'interests': interests},
                    'speaker': 'Visitor', 'message': message, 'kind': 'chat', 'nearby_names': [name, other],
                    'recent_turns': [], 'conversation_with_candidate': False, 'speaker_hired_candidate': True}
            variants = []
            variants.append(('hired_opening', 'join', deepcopy(base)))
            bystander = deepcopy(base)
            bystander['speaker_hired_candidate'] = False
            variants.append(('unaddressed_bystander', 'quiet', bystander))
            for hired in (True, False):
                current = deepcopy(base)
                current['speaker_hired_candidate'] = hired
                current['conversation_with_candidate'] = True
                current['recent_turns'] = [
                    {'speaker': 'Visitor', 'text': f'{name}, could we spend a moment talking together?'},
                    {'speaker': name, 'text': 'Of course. I would be happy to listen and get to know you.'}]
                variants.append(('hired_ongoing' if hired else 'nonhired_ongoing', 'join', current))
            addressed = deepcopy(base)
            addressed['speaker_hired_candidate'] = False
            addressed['message'] = f'{name}, {message}'
            variants.append(('explicit_address', 'join', addressed))
            elsewhere = deepcopy(base)
            elsewhere['message'] = f'{other}, {message}'
            variants.append(('other_addressee', 'quiet', elsewhere))
            quoted = deepcopy(base)
            quoted['message'] = f'{other}, the note says "{name}, {message}". Is that copied accurately?'
            variants.append(('quoted_address', 'quiet', quoted))
            silence = deepcopy(base)
            silence['message'] = f'Please do not answer yet; I am rehearsing this line: "{message}"'
            variants.append(('requested_silence', 'quiet', silence))
            family = f'v3-hired-opening-{split}-{index:02d}'
            for scenario, label, state in variants:
                rows.append({'id': f'participation-{family}-{scenario}', 'family_id': family,
                             'split_group': family, 'split': split, 'label': label, 'state': state,
                             'origin': 'assistant_authored_synthetic', 'review_status': 'synthetic_draft',
                             'scenario': scenario})  # Variants stay together; no model output supplies a gold label.
    return rows


def validate(repo, rows):
    sys.path[:0] = [str(repo), str(repo / 'python')]
    from participation_data import validate_rows
    validate_rows(rows, ('train', 'validation', 'development', 'test'))
    source = repo / 'data/participation'
    previous = read_rows(source / 'synthetic-expanded-v2.jsonl') + read_rows(source / 'development-v2.jsonl')
    actual = {row['id']: row for row in rows}
    if any(actual.get(row['id']) != row for row in previous):
        raise ValueError('Every v2 row and split must remain unchanged')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    target = args.output or args.repo / 'data/participation'
    rows = build(args.repo)
    validate(args.repo, rows)
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, selected in (('synthetic-expanded-v3.jsonl', [row for row in rows if row['split'] != 'development']),
                           ('development-v3.jsonl', [row for row in rows if row['split'] == 'development'])):
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected)
        path = target / name
        if path.exists() and path.read_text(encoding='utf-8') != content:
            raise ValueError('Refusing to overwrite changed dataset: ' + str(path))
        path.write_text(content, encoding='utf-8', newline='\n')
        hashes[name] = hashlib.sha256(content.encode()).hexdigest()
    new = [row for row in rows if row['id'].startswith('participation-v3-')]
    manifest = {'task': 'participation', 'total_rows': len(rows),
                'splits': dict(Counter(row['split'] for row in rows)),
                'new_rows_by_split': dict(Counter(row['split'] for row in new)),
                'new_families_by_split': {split: len(messages) for split, messages in SCENARIOS.items()},
                'sha256_utf8_lf': hashes, 'preserves_v2_rows_and_splits': True,
                'review_status': 'synthetic_draft', 'real_exported_decisions': 0,
                'purpose': 'Hired/no-history implicit social address, bounded by explicit addressee and silence/quotation negatives',
                'limitations': ['Context variants within a family are correlated.',
                               'The reported live failure informed this targeted expansion.',
                               'New authored holdouts share context templates; they are not real-player validation.',
                               'Seeded ramp replay is a regression set only and is excluded from training.']}
    (target / 'manifest-v3.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
