"""Merge a selected LoRA into its pinned unquantized base on CPU, without changing production."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_run(metadata, completion, task):
    from train_classifier import task_schema
    if task == 'multitask':
        from multitask_data import POLICY, TASKS, LAYOUTS
        if (metadata.get('task') != task or metadata.get('policy') != POLICY
                or not completion['completed'] or completion['smoke_test']):
            raise ValueError('Merge requires a completed full multi-task run')
        if set(metadata.get('tasks', {})) != set(TASKS):
            raise ValueError('Multi-task metadata must describe both tasks')
        for subtask in TASKS:
            expected = task_schema(subtask, LAYOUTS[subtask])
            if any(metadata['tasks'][subtask].get(key) != value for key, value in expected.items()):
                raise ValueError('Multi-task training schema does not match the selected task')
        return {'task': task, 'policy': POLICY}  # Merge still requires explicit multi-task selection.
    expected = task_schema(task, metadata.get('prompt_layout', 'state_first'))
    if (metadata.get('task', 'routing') != task or metadata['policy'] != expected['policy']
            or not completion['completed'] or completion['smoke_test']):
        raise ValueError('Merge requires a completed full run with the selected task and policy')
    if metadata.get('question') != expected['question'] or metadata.get('options') != expected['options']:
        raise ValueError('Training question/options do not match the selected task')
    return expected  # Explicit task selection prevents exporting participation weights as the routing release.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--task', choices=('routing', 'participation', 'multitask'), default='routing')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    output = args.output or args.run / 'merged'
    if output.exists():
        raise ValueError('Merge output already exists; choose a fresh directory')
    metadata = json.loads((args.run / 'run.json').read_text())
    completion = json.loads((args.run / 'completion.json').read_text())
    validate_run(metadata, completion, args.task)
    if args.local_files_only:
        os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(metadata['model'], revision=metadata['revision'],
                cache_dir=ROOT / 'data/models/training', dtype=torch.bfloat16, device_map={'': 'cpu'}, local_files_only=args.local_files_only)
    model = PeftModel.from_pretrained(model, args.run / 'best-adapter')
    model = model.merge_and_unload(safe_merge=True)  # Merge into original BF16 weights, not dequantized NF4 approximations.
    model.save_pretrained(output, safe_serialization=True)
    for path in (args.run / 'best-adapter').iterdir():
        if path.name not in ('adapter_config.json', 'adapter_model.safetensors', 'README.md'):
            shutil.copy2(path, output / path.name)
    (output / 'merge_provenance.json').write_text(json.dumps({
        'run': str(args.run.resolve()), 'base_revision': metadata['revision'],
        'selected_epoch': completion['selected_epoch'], 'task': args.task, 'policy': metadata['policy'], 'quantized': False,
        'deployment_ready': False}, indent=2) + '\n')
    print('Merged checkpoint:', output, flush=True)


if __name__ == '__main__':
    main()
