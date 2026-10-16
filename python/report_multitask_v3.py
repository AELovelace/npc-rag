"""Compare v2/v3 on retained tests, new context families and reconstructed opening regressions."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def summary(rows):
    scored = [row for row in rows if row.get('expected') is not None]
    return {'total': len(scored), 'correct': sum(row['predicted'] == row['expected'] for row in scored),
            'unwanted_joins': sum(row['expected'] == 'quiet' and row['predicted'] == 'join' for row in scored),
            'missed_joins': sum(row['expected'] == 'join' and row['predicted'] == 'quiet' for row in scored)}


def replay(release, gguf, cases):
    from evaluate_multitask import validate_release
    validate_release(release)
    import llama_cpp
    from decider.engine_gguf import GGUFEngine
    from npc_rag.native_logging import install_native_error_logging
    from npc_rag.participation_policy import build_participation_item
    install_native_error_logging(llama_cpp)
    engine = GGUFEngine(str(release / gguf), str(release), n_gpu_layers=0, n_threads=4, n_ctx=1024, n_seq_max=1, verbose=True)
    output = []
    try:
        for case in cases:
            item = build_participation_item(case['state'], engine.m.tok, 768)
            probabilities = engine.score_items([item], temperature=1.0)[0][0][:2].tolist()
            output.append({'id': case['id'], 'cohort': case['cohort'], 'expected': case['label'],
                           'predicted': 'join' if probabilities[0] >= probabilities[1] else 'quiet',
                           'join_probability': probabilities[0]})
        return output
    finally:
        engine.__del__()
        engine.L = None  # Release one model before loading the other; never double-free native pointers.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--baseline-release', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    args = parser.parse_args()
    current = read_json(args.release / 'quantized-multitask.json')
    previous = read_json(args.run / 'previous-model-on-current-data.json')
    if current['evaluation_sources'] != previous['evaluation_sources']:
        raise ValueError('Both models must be evaluated on the exact same sources')
    new_rows = current['datasets']['synthetic_test']['rows']
    old_rows = previous['datasets']['synthetic_test']['rows']
    if {(row['id'], row['task'], row['expected']) for row in old_rows} != {(row['id'], row['task'], row['expected']) for row in new_rows}:
        raise ValueError('Evaluation cases disagree')
    cohorts = {'routing': lambda row: row['task'] == 'routing',
               'preserved_participation_v2': lambda row: row['task'] == 'participation' and not row['id'].startswith('participation-v3-'),
               'new_v3': lambda row: row['id'].startswith('participation-v3-')}
    dataset = Path(current['evaluation_sources']['participation']['dataset'])
    sources = {row['id']: row for row in (json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines())}
    for scenario in sorted({row['scenario'] for row in sources.values() if row['id'].startswith('participation-v3-')}):
        cohorts['v3_' + scenario] = lambda row, selected=scenario: row['id'].startswith('participation-v3-') and sources[row['id']]['scenario'] == selected
    comparisons = {name: {'v2': summary([row for row in old_rows if include(row)]),
                           'v3': summary([row for row in new_rows if include(row)])} for name, include in cohorts.items()}
    cases = [json.loads(line) for line in args.replay.read_text(encoding='utf-8').splitlines()]
    regression = {}
    for name, release, gguf in (('v2', args.baseline_release, 'decider-multitask-v2.gguf'),
                                ('v3', args.release, 'decider-multitask-v3.gguf')):
        print('Replaying saved openings against ' + name, flush=True)
        rows = replay(release, gguf, cases)
        regression[name] = {'summary': summary(rows), 'rows': rows,
                            'unlabelled_bystander_joins': sum(row['cohort'] == 'bystander_opening' and row['predicted'] == 'join' for row in rows)}
    gates = {
        'routing_no_regression': comparisons['routing']['v3']['correct'] >= comparisons['routing']['v2']['correct'],
        'preserved_participation_no_regression': comparisons['preserved_participation_v2']['v3']['correct'] >= comparisons['preserved_participation_v2']['v2']['correct'],
        'new_hired_openings_all_join': comparisons['v3_hired_opening']['v3']['missed_joins'] == 0,
        'seeded_hired_social_openings_all_join': regression['v3']['summary']['total'] > 0 and regression['v3']['summary']['missed_joins'] == 0,
        'explicit_boundaries_no_false_join': all(comparisons['v3_' + name]['v3']['unwanted_joins'] == 0 for name in ('other_addressee', 'quoted_address', 'requested_silence')),
        'ongoing_conversations_all_join': all(comparisons['v3_' + name]['v3']['missed_joins'] == 0 for name in ('hired_ongoing', 'nonhired_ongoing')),
        'unaddressed_bystanders_stay_quiet': comparisons['v3_unaddressed_bystander']['v3']['unwanted_joins'] == 0,
        'fixed_raw_routing_no_regression': current['datasets']['fixed_routing_raw_model']['accuracy'] >= previous['datasets']['fixed_routing_raw_model']['accuracy'],
    }
    output = {'comparisons': comparisons, 'opening_regression': regression, 'gates': gates,
              'offline_gates_pass': all(gates.values()), 'deployed': False,
              'replay_sha256': hashlib.sha256(args.replay.read_bytes()).hexdigest(),
              'limitations': read_json(Path(str(args.replay) + '.manifest.json'))['limitations']}
    (args.run / 'targeted-comparison.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'comparisons': comparisons, 'gates': gates}, indent=2))
    # Failed behavior gates remain visible in the report; never silently deploy or retrain on test failures.


if __name__ == '__main__':
    main()
