"""Compare the completed combined adapter against the two saved specialist test reports."""
import argparse
import hashlib
import json
from pathlib import Path

from multitask_data import TASKS


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_participation_expansion(current_source, previous_meta):
    previous_dataset = Path(previous_meta['dataset'])
    previous_development = Path(previous_meta.get('development', previous_dataset.parent / 'development-v1.jsonl'))
    for old_path, new_path, hash_key in (
            (previous_dataset, Path(current_source['dataset']), 'dataset_sha256'),
            (previous_development, Path(current_source['development']), 'development_sha256')):
        if hashlib.sha256(old_path.read_bytes()).hexdigest() != previous_meta[hash_key]:
            raise ValueError('Previous participation source changed; cannot compare preserved holdouts')
        if hashlib.sha256(new_path.read_bytes()).hexdigest() != current_source[hash_key]:
            raise ValueError('Expanded participation source changed since training')
        old = [json.loads(line) for line in old_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        new = {row['id']: row for row in (json.loads(line) for line in new_path.read_text(encoding='utf-8').splitlines() if line.strip())}
        if any(new.get(row['id']) != row for row in old):
            raise ValueError('Expansion must preserve all original participation rows, labels and splits')
    # Changed source hashes are allowed only after proving the prior examples remain byte-equivalent as JSON values.


def compare(run, specialists, *, allow_expanded_participation=False):
    from merge_classifier import validate_run
    metadata = read_json(run / 'run.json')
    validate_run(metadata, read_json(run / 'completion.json'), 'multitask')
    final = read_json(run / 'final-test.json')
    comparisons = {}
    for task in TASKS:
        source = specialists[task]
        previous_meta = read_json(source / 'run.json')
        validate_run(previous_meta, read_json(source / 'completion.json'), task)
        current_source = metadata['tasks'][task]
        expanded = (task == 'participation' and allow_expanded_participation
                    and any(current_source[key] != previous_meta[key] for key in ('dataset_sha256', 'development_sha256')))
        if expanded:
            validate_participation_expansion(current_source, previous_meta)
        for key in ('dataset_sha256', 'development_sha256', 'policy', 'question', 'options'):
            if expanded and key in ('dataset_sha256', 'development_sha256'):
                continue
            if current_source[key] != previous_meta[key]:
                raise ValueError(f'{task}: mismatched {key}; rerun the specialist on the same inputs')
        if any(metadata[key] != previous_meta[key] for key in ('model', 'revision')):
            raise ValueError(f'{task}: different base model/revision')
        previous = read_json(source / 'final-test.json')
        current = final['by_task'][task]
        current_rows = [row for row in final['rows'] if row['task'] == task]
        if expanded:
            from train_classifier import report_rows
            previous_ids = {row['id'] for row in previous['rows']}
            current_rows = [row for row in current_rows if row['id'] in previous_ids]
            current = report_rows(current_rows, task=task)
        if sorted((row['id'], row['expected']) for row in previous['rows']) != sorted((row['id'], row['expected']) for row in current_rows):
            raise ValueError(f'{task}: test IDs or labels differ')
        recalls = {label: values['recall'] - previous['by_class'][label]['recall'] for label, values in current['by_class'].items()}
        comparisons[task] = {
            'specialist_run': str(source.resolve()), 'total': current['total'],
            'specialist_accuracy': previous['accuracy'], 'multitask_accuracy': current['accuracy'],
            'accuracy_delta': current['accuracy'] - previous['accuracy'], 'recall_deltas': recalls,
            'no_test_regression': current['accuracy'] >= previous['accuracy'] and all(delta >= 0 for delta in recalls.values()),
            'multitask': current,
            'comparison_partition': 'preserved v1 test subset' if expanded else 'entire unchanged test set',
            'expanded_test_summary': final['by_task'][task] if expanded else None,
        }
    return {'scope': 'Saved NF4 QLoRA adapter reports on identical preserved synthetic test examples; expanded results reported separately; not quantized/live qualification',
            'tasks': comparisons, 'both_tasks_without_test_regression': all(item['no_test_regression'] for item in comparisons.values()),
            'deployment_ready': False}  # A passing synthetic comparison does not calibrate confidence or validate a shared worker pool.


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--routing-run', type=Path, default=root / 'runs/classifier-v3-first-pass')
    parser.add_argument('--participation-run', type=Path, default=root / 'runs/participation-v1-first-pass-r2-20261014')
    parser.add_argument('--allow-expanded-participation', action='store_true', help='Compare only preserved original participation test rows after auditing the expansion')
    args = parser.parse_args()
    report = compare(args.run, {'routing': args.routing_run, 'participation': args.participation_run},
                     allow_expanded_participation=args.allow_expanded_participation)
    (args.run / 'specialist-comparison.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
