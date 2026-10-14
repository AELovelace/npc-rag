"""Merge a selected LoRA into its pinned unquantized base on CPU, without changing production."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npc_rag.decider_policy import policy_for_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.run / 'merged'
    if output.exists():
        raise ValueError('Merge output already exists; choose a fresh directory')
    metadata = json.loads((args.run / 'run.json').read_text())
    completion = json.loads((args.run / 'completion.json').read_text())
    if metadata['policy'] != policy_for_layout(metadata.get('prompt_layout', 'state_first')) or not completion['completed'] or completion['smoke_test']:
        raise ValueError('Merge requires a completed full run with the current policy')
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(metadata['model'], revision=metadata['revision'],
                cache_dir=ROOT / 'data/models/training', dtype=torch.bfloat16, device_map={'': 'cpu'})
    model = PeftModel.from_pretrained(model, args.run / 'best-adapter')
    model = model.merge_and_unload(safe_merge=True)  # Merge into original BF16 weights, not dequantized NF4 approximations.
    model.save_pretrained(output, safe_serialization=True)
    for path in (args.run / 'best-adapter').iterdir():
        if path.name not in ('adapter_config.json', 'adapter_model.safetensors', 'README.md'):
            shutil.copy2(path, output / path.name)
    (output / 'merge_provenance.json').write_text(json.dumps({
        'run': str(args.run.resolve()), 'base_revision': metadata['revision'],
        'selected_epoch': completion['selected_epoch'], 'quantized': False,
        'deployment_ready': False}, indent=2) + '\n')
    print('Merged checkpoint:', output, flush=True)


if __name__ == '__main__':
    main()
