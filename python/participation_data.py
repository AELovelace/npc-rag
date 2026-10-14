"""Validate context-rich training rows and prevent conversation-family leakage."""
import json
from collections import defaultdict
from npc_rag.participation_policy import LABELS, encode_state


def message_key(row):
    return ' '.join(row['state']['message'].casefold().split())


def validate_rows(rows, allowed_splits):
    if not rows:
        raise ValueError('Participation dataset is empty')
    seen, states, groups, families, messages, splits = set(), set(), {}, {}, {}, defaultdict(list)
    for row in rows:
        if row.get('label') not in LABELS or row.get('split') not in allowed_splits:
            raise ValueError('Unknown participation label or split')
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ('id', 'family_id', 'split_group')):
            raise ValueError('Each participation row needs ID, family_id and split_group')
        state_key = encode_state(row['state'])
        if row['id'] in seen or state_key in states:
            raise ValueError('Duplicate participation ID or complete context')
        seen.add(row['id']); states.add(state_key)
        for mapping, key in ((groups, row['split_group']), (families, row['family_id']), (messages, message_key(row))):
            if key in mapping and mapping[key] != row['split']:
                raise ValueError('Related participation examples cross data splits')
            mapping[key] = row['split']
        splits[row['split']].append(row)
    if any({r['label'] for r in splits[s]} != set(LABELS) for s in allowed_splits):
        raise ValueError('Every participation split must contain both labels')
    return dict(splits)  # Identical messages may have opposite labels when context changes, within one split only.


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def load_dataset(path):
    return validate_rows(read_rows(path), ('train', 'validation', 'test'))


def load_development(path, splits):
    rows = read_rows(path)
    validate_rows(rows, ('development',))
    combined = [row for group in splits.values() for row in group] + rows
    validate_rows(combined, ('train', 'validation', 'test', 'development'))
    return rows  # Checkpoint-selection conversations must also be separate from the final test.
