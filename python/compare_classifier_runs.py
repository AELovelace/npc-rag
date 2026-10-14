"""Produce a matched final qualification report; never choose or retrain a checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path


def compare(base, tuned):
    for key in ('policy', 'evidence_url', 'wiki_evidence', 'keyword_coverage', 'keyword_dense', 'raw_model', 'wiki_only'):
        if base[key] != tuned[key]:
            raise ValueError(f'Unmatched evaluation configuration: {key}')
    for key in ('decision_policy', 'engine'):
        if base['classifier'][key] != tuned['classifier'][key]:
            raise ValueError(f'Unmatched model policy/format: {key}')
    if base['raw_model'] or base['wiki_only']:
        raise ValueError('Final qualification requires full hybrid evaluation')
    if len(base['results']) != len(tuned['results']):
        raise ValueError('Different numbers of evaluation cases')
    for old, new in zip(base['results'], tuned['results']):
        if any(old[k] != new[k] for k in ('message', 'expected', 'dataset')):
            raise ValueError('Different evaluation cases/order')
        if old['method'] == 'error' or new['method'] == 'error':
            raise ValueError('Infrastructure errors invalidate the matched comparison')
        for key in ('wiki_best', 'wiki_keyword'):
            if old.get('scores', {}).get(key) != new.get('scores', {}).get(key):
                raise ValueError('Wiki evidence changed between model runs')
    def metrics(report):
        summary = report['summary']
        return {'accuracy': summary['accuracy'],
                'game_recall': summary['by_class']['game']['accuracy'],
                'chat_recall': summary['by_class']['general']['accuracy'],
                'game_missed_as_chat': summary['game_missed_as_chat'],
                'chat_sent_to_game': summary['chat_sent_to_game']}
    before, after = metrics(base), metrics(tuned)
    improves = after['accuracy'] > before['accuracy'] and min(after['game_recall'], after['chat_recall']) >= min(before['game_recall'], before['chat_recall'])
    def target(report):
        heldout, mixed = report['datasets']['held_out'], report['datasets']['mixed_intent']
        if heldout['total'] != 80 or mixed['total'] != 16:
            raise ValueError('Qualification requires all 80 held-out and 16 mixed cases')
        return {'held_out_accuracy': heldout['accuracy'], 'mixed_accuracy': mixed['accuracy'],
                'passed': heldout['accuracy'] >= .9375 and mixed['accuracy'] == 1.0,
                'target': 'At least 75/80 held-out (93.8% rounded) and 16/16 mixed (100%)'}
    return {'base': before, 'fine_tuned': after, 'held_out_improved': improves,
            'qwen_target': {'base': target(base), 'fine_tuned': target(tuned)},
            'improvement_rule': 'Higher total accuracy without lower worst-class recall on the identical final hybrid suite',
            'both_classes_pass_90_percent': min(after['game_recall'], after['chat_recall']) >= .9,
            'wiki_evidence_identical': True, 'policy': base['policy'],
            'decision_policy': base['classifier']['decision_policy'], 'engine': base['classifier']['engine']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--tuned', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.base.read_text()), json.loads(args.tuned.read_text()))
    result['training'] = json.loads((args.run / 'completion.json').read_text())
    result['provenance'] = json.loads((args.run / 'run.json').read_text())
    result['artifacts'] = {key: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                           for key, path in (('base', args.base), ('fine_tuned', args.tuned))}
    result['limitations'] = ['Synthetic labels have not had independent human review.',
                            'The development set selected the checkpoint; final evaluation did not.',
                            'Merged BF16 evaluation does not qualify a Q4_K_M export.',
                            'Fine-tuned probabilities use temperature 1 and are explicitly uncalibrated.',
                            'Previous NPC context is not included. No production deployment was performed.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('provenance', 'artifacts')}, indent=2))


if __name__ == '__main__':
    main()
