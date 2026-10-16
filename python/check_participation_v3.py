"""Audit all task prompts and exclude the seeded regression messages before training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    folder = args.data or args.repo / 'data/participation'
    sys.path[:0] = [str(args.repo), str(args.repo / 'python')]
    from multitask_data import load_multitask, BUDGETS, LAYOUTS, loss_weights
    from train_classifier import encode_rows
    from build_classifier_seed import normalize
    from transformers import AutoTokenizer
    splits, _ = load_multitask(SimpleNamespace(dataset=None, dev_dataset=None,
                  participation_dataset=folder / 'synthetic-expanded-v3.jsonl',
                  participation_dev_dataset=folder / 'development-v3.jsonl'))
    replay = [json.loads(line) for line in (folder / 'seeded-opening-regression-v3.jsonl').read_text(encoding='utf-8').splitlines()]
    reserved = {normalize(row['state']['message']) for row in replay}
    for row in splits['train']:
        if normalize(row.get('message', row.get('state', {}).get('message', ''))) in reserved:
            raise ValueError('A seeded regression message leaked into training')
    tokenizer = AutoTokenizer.from_pretrained(str(args.repo / 'runs/releases/decider-multitask-v2-q4km'), local_files_only=True)
    lengths = {}
    for task in BUDGETS:
        rows = [row for split in splits.values() for row in split if row['task'] == task]
        encoded = encode_rows(rows, tokenizer, BUDGETS[task], layout=LAYOUTS[task], task=task)
        lengths[task] = max(len(row['ids']) for row in encoded)
    print(json.dumps({'split_counts': {split: dict(Counter(row['task'] for row in rows)) for split, rows in splits.items()},
                      'max_prompt_tokens': lengths, 'loss_weights': loss_weights(splits['train']),
                      'seeded_regression_rows': len(replay), 'regression_training_overlap': 0}, indent=2))


if __name__ == '__main__':
    main()
