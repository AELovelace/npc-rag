"""Evaluate both tasks sequentially through one local experimental GGUF model instance."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multitask_data import load_multitask, POLICY, TASKS, LAYOUTS, BUDGETS
from train_classifier import encode_rows, report_rows, task_labels


def validate_release(release):
    from merge_classifier import validate_run
    metadata = json.loads((release / 'routing_policy.json').read_text(encoding='utf-8'))
    validate_run(metadata, {'completed': True, 'smoke_test': metadata['smoke_test']}, 'multitask')
    config = json.loads((release / 'decider_config.json').read_text(encoding='utf-8'))
    if config.get('routing_policy') != POLICY or config.get('task_prompt_layouts') != LAYOUTS or config.get('layout') != 'plain':
        raise ValueError('Unexpected multi-task checkpoint config')
    return metadata  # Verify both schemas before native model allocation.


def evaluate_engine(engine, rows):
    results = []
    for row in rows:
        started = time.perf_counter()
        task = row['task']
        item = encode_rows([row], engine.m.tok, BUDGETS[task], layout=LAYOUTS[task], task=task)[0]
        # Score one real prompt at a time using the same two A/B slots as adapter evaluation.
        probs = engine.score_items([item], temperature=1.0)[0][0][:2].tolist()
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs) or not math.isclose(sum(probs), 1, abs_tol=1e-4):
            raise ValueError('Invalid native decision probabilities')
        predicted = task_labels(task)[max(range(2), key=lambda index: probs[index])]
        results.append({'id': row['id'], 'task': task, 'expected': row['label'], 'predicted': predicted,
                        'probabilities_by_option_order': probs, 'ms': round((time.perf_counter() - started) * 1000, 2)})
        if len(results) % 16 == 0:
            print(f'Evaluated {len(results)}/{len(rows)}', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--gguf-file', default='decider-multitask-v1.gguf')
    args = parser.parse_args()
    metadata = validate_release(args.release)
    sources = metadata['tasks']
    splits, current_sources = load_multitask(SimpleNamespace(
        dataset=Path(sources['routing']['dataset']), dev_dataset=Path(sources['routing']['development']),
        participation_dataset=Path(sources['participation']['dataset']),
        participation_dev_dataset=Path(sources['participation']['development'])))
    for task in TASKS:
        if any(current_sources[task][key] != sources[task][key] for key in ('dataset_sha256', 'development_sha256')):
            raise ValueError('Evaluation data changed since training')
    import llama_cpp
    from decider.engine_gguf import GGUFEngine
    from npc_rag.native_logging import install_native_error_logging
    install_native_error_logging(llama_cpp)
    engine = GGUFEngine(str(args.release / args.gguf_file), str(args.release),
                        n_gpu_layers=0, n_threads=4, n_ctx=1024, n_seq_max=1, verbose=True)
    try:
        results = {}
        # Alternate tasks to exercise changes of question and prompt layout with a single loaded model.
        task_rows = {task: [row for row in splits['test'] if row['task'] == task] for task in TASKS}
        from itertools import zip_longest
        alternating = [row for pair in zip_longest(*task_rows.values()) for row in pair if row is not None]
        rows = evaluate_engine(engine, alternating)
        results['synthetic_test'] = {**report_rows(rows, task='multitask'), 'rows': rows}
        fixed = []
        for name in ('eval_messages.json', 'eval_mixed_messages.json'):
            data = json.loads((ROOT / 'data' / name).read_text(encoding='utf-8'))
            fixed.extend({'task': 'routing', 'id': f'{name}/{label}/{index}', 'message': message, 'label': label}
                         for label, messages in data.items() for index, message in enumerate(messages))
        fixed_rows = evaluate_engine(engine, fixed)
        results['fixed_routing_raw_model'] = {**report_rows(fixed_rows), 'rows': fixed_rows}
        report = {'policy': POLICY, 'quantization': 'Q4_K_M', 'loaded_model_instances': 1,
                  'calibration': 'temperature 1; uncalibrated', 'datasets': results, 'deployment_ready': False,
                  'scope': 'Sequential local CPU evaluation; fixed routing is raw model without wiki evidence; no live shared-pool test'}
        (args.release / 'quantized-multitask.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({key: {k: v for k, v in value.items() if k != 'rows'} for key, value in results.items()}, indent=2))
    finally:
        engine.__del__()  # Release the native context/model even if inference or report writing raises.
        engine.L = None  # The upstream destructor does not clear pointers; prevent a second free at collection.


if __name__ == '__main__':
    main()
