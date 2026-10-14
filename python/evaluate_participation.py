"""Evaluate the exported participation GGUF with the exact production prompt path."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npc_rag.config import load_settings
from npc_rag.participation import ParticipationClassifier
from participation_data import load_dataset, load_development
from train_classifier import report_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    splits = load_dataset(ROOT / 'data/participation/synthetic-pilot-v1.jsonl')
    development = load_development(ROOT / 'data/participation/development-v1.jsonl', splits)
    backend = ParticipationClassifier(replace(load_settings(), participation_model=str(args.release.resolve()),
                                              decider_gpu_layers=0, decider_threads=4))
    results = {}
    try:
        backend._load()
        for split, examples in (('development', development), ('test', splits['test'])):
            rows = []
            for row in examples:
                began = time.perf_counter()
                decision = backend._decide(row['state'])
                rows.append({'id': row['id'], 'expected': row['label'], 'predicted': decision.label,
                             'join_probability': decision.join_probability,
                             'ms': round((time.perf_counter() - began) * 1000, 2)})
                if len(rows) % 8 == 0:
                    print(f'{split}: {len(rows)}/{len(examples)}', flush=True)
            results[split] = {**report_rows(rows, task='participation'), 'families': len({r['family_id'] for r in examples}), 'rows': rows}
        report = {'model': backend.view(), 'quantization': 'Q4_K_M', 'datasets': results,
                  'calibration': 'temperature 1; uncalibrated after training/quantization',
                  'latency_scope': 'Sequential local CPU inference; not server throughput or a Qwen A/B',
                  'deployment_ready': False}
        (args.release / 'quantized-participation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({key: {k: v for k, v in value.items() if k != 'rows'} for key, value in results.items()}, indent=2))
    finally:
        backend._unload()
        backend._worker.shutdown(wait=True)  # Evaluation never leaves native buffers or executor threads behind.


if __name__ == '__main__':
    main()
