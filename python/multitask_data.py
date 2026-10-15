"""Prepare the two Decider tasks without mixing their labels or held-out partitions."""
from collections import Counter
import hashlib

TASKS = ('routing', 'participation')
LAYOUTS = {'routing': 'state_first', 'participation': 'schema_first'}
BUDGETS = {'routing': 384, 'participation': 768}
POLICY = 'routing-participation-multitask-v1'


def load_fixed_routing():
    import json
    from train_classifier import ROOT, LABELS
    rows = []
    for name in ('eval_messages.json', 'eval_mixed_messages.json'):
        data = json.loads((ROOT / 'data' / name).read_text(encoding='utf-8'))
        rows.extend({'task': 'routing', 'id': f'{name}/{label}/{index}', 'message': message, 'label': label}
                    for label in LABELS for index, message in enumerate(data[label]))
    return rows  # Read only known label arrays; descriptive _about metadata is not an evaluation example.


def load_multitask(args):
    from train_classifier import ROOT, load_dataset, load_development, task_schema
    paths = {
        'routing': (args.dataset or ROOT / 'data/classifier/synthetic-pilot-v2.jsonl',
                    args.dev_dataset or ROOT / 'data/classifier/realistic-dev-v2.jsonl'),
        'participation': (args.participation_dataset or ROOT / 'data/participation/synthetic-pilot-v1.jsonl',
                          args.participation_dev_dataset or ROOT / 'data/participation/development-v1.jsonl'),
    }
    combined, sources = {}, {}
    for task, (dataset, development) in paths.items():
        splits = load_dataset(dataset, task=task)
        splits['development'] = load_development(development, splits, task=task)
        for split, rows in splits.items():
            combined.setdefault(split, []).extend({**row, 'task': task} for row in rows)
        sources[task] = {
            **task_schema(task, LAYOUTS[task]), 'prompt_layout': LAYOUTS[task],
            'max_tokens': BUDGETS[task], 'dataset': str(dataset.resolve()),
            'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'development': str(development.resolve()),
            'development_sha256': hashlib.sha256(development.read_bytes()).hexdigest(),
            'split_counts': {split: len(rows) for split, rows in splits.items()},
        }
    # Detect a held-out utterance reused by the other task, including fixed routing evaluation.
    from build_classifier_seed import normalize
    held_out = {normalize(row['message']) for row in load_fixed_routing()}
    for split, rows in combined.items():
        if split != 'train':
            held_out.update(normalize(row.get('message', row.get('state', {}).get('message', ''))) for row in rows)
    for row in combined['train']:
        message = row.get('message', row.get('state', {}).get('message', ''))
        if normalize(message) in held_out:
            raise ValueError('Multi-task training message overlaps a held-out message')
    return combined, sources  # Validate before loading GPU packages or allocating model memory.


def select_rows(splits, smoke_test):
    if not smoke_test:
        return splits
    return {split: [row for task in TASKS for row in [r for r in rows if r['task'] == task][:8]]
            for split, rows in splits.items()}  # Exercise both prompt paths during the real GPU smoke test.


def loss_weights(rows):
    counts = Counter(row['task'] for row in rows)
    if set(counts) != set(TASKS):
        raise ValueError('Multi-task training requires both tasks')
    return {task: len(rows) / (len(TASKS) * counts[task]) for task in TASKS}


def augment_multitask(rows, seed, smoke_test=False):
    from classifier_augmentation import augment_rows, augmentation_stats
    routing = augment_rows([row for row in rows if row['task'] == 'routing'], seed)
    participation = [row for row in rows if row['task'] == 'participation']
    stats = {'routing': {'smoke_subset': True} if smoke_test else augmentation_stats(routing),
             'participation': {'augmented': False}, 'loss_weights': loss_weights(rows)}
    return routing + participation, stats  # Only routing messages receive the existing typo/style augmentation.


def report_multitask(rows):
    from train_classifier import report_rows
    reports = {task: report_rows([row for row in rows if row['task'] == task], task=task) for task in TASKS}
    classes = {f'{task}/{label}': value for task, report in reports.items() for label, value in report['by_class'].items()}
    return {'total': len(rows), 'accuracy': sum(report['accuracy'] for report in reports.values()) / len(TASKS),
            'accuracy_aggregation': 'equal task macro average', 'by_class': classes,
            'gate_passed': all(report['gate_passed'] for report in reports.values()), 'by_task': reports}


def checkpoint_config():
    return {'routing_policy': POLICY, 'task': 'multitask',
            'task_prompt_layouts': LAYOUTS, 'calibration_status': 'uncalibrated_after_finetuning'}
    # A distinct policy prevents old single-task loaders from silently accepting these experimental weights.
