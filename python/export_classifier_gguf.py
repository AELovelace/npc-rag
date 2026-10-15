"""Export the selected classifier through a pinned llama.cpp converter and native Q4_K_M quantizer."""
import argparse
import ctypes
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
LLAMA_CPP_REVISION = '0c1e57098bba43ac29e6e3b677cdceebdd22334f'
SUPPORT_FILES = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                 'decider_config.json', 'routing_policy.json', 'config.json', 'merge_provenance.json')


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()  # Stream multi-GB weights rather than loading them into RAM.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/classifier-v3-first-pass')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/releases/decider0-1-q4km')
    parser.add_argument('--converter-root', type=Path, default=ROOT / 'runs/gguf-export-tools' / LLAMA_CPP_REVISION)
    parser.add_argument('--task', choices=('routing', 'participation', 'multitask'), default='routing')
    parser.add_argument('--name', default='decider0-1')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', args.name):
        raise ValueError('Release name must contain lowercase letters, digits and hyphens only')
    if args.task != 'routing' and args.name == 'decider0-1':
        raise ValueError('Participation/multi-task export requires its own release name')
    from merge_classifier import validate_run
    metadata = json.loads((args.run / 'run.json').read_text())
    validate_run(metadata, json.loads((args.run / 'completion.json').read_text()), args.task)
    merged = args.run / 'merged'
    provenance = json.loads((merged / 'merge_provenance.json').read_text())
    if provenance.get('task', 'routing') != args.task or provenance.get('quantized') or not (merged / 'model.safetensors').is_file():
        raise ValueError('Expected the selected, merged BF16 checkpoint')
    if args.output.exists():
        raise ValueError('Output already exists; choose a fresh export directory')
    import llama_cpp
    if version('llama-cpp-python') != '0.3.36':
        raise ValueError('This export is pinned to llama-cpp-python 0.3.36')
    args.output.mkdir(parents=True)
    intermediate = args.output / (args.name + '-bf16.gguf')
    destination = args.output / (args.name + '.gguf')
    log = args.output / 'conversion.log'
    command = [sys.executable, str(args.converter_root / 'convert_hf_to_gguf.py'), str(merged),
               '--outtype', 'bf16', '--no-mtp', '--outfile', str(intermediate), '--model-name', 'LiDollQuest ' + args.name]
    # HF retains an MTP-layer count but these classifier weights have no MTP tensors; omit that unused draft layer.
    print('Converting selected BF16 weights to GGUF...', flush=True)
    with log.open('w', encoding='utf-8') as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join(log.read_text(encoding='utf-8').splitlines()[-25:]))
        raise RuntimeError(f'Conversion failed; see {log}')
    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M
    params.nthread = 4
    print('Quantizing GGUF to Q4_K_M...', flush=True)
    llama_cpp.llama_backend_init()
    result = llama_cpp.llama_model_quantize(os.fsencode(intermediate), os.fsencode(destination), ctypes.byref(params))
    if result != 0 or not destination.is_file():
        raise RuntimeError(f'Quantization failed with code {result}')
    for name in SUPPORT_FILES:
        shutil.copy2(merged / name, args.output / name)
    manifest = {'name': args.name, 'task': args.task, 'policy': metadata['policy'], 'quantization': 'Q4_K_M',
                'download_url': 'https://lidoll.dev/gguf/' + args.name + '.gguf',
                'llama_cpp_revision': LLAMA_CPP_REVISION, 'llama_cpp_python': version('llama-cpp-python'),
                'conversion_arguments': command[2:], 'mtp_excluded': True,
                'source_weights_sha256': sha256(merged / 'model.safetensors'),
                'converter_sha256': sha256(args.converter_root / 'convert_hf_to_gguf.py'),
                'source_archive_sha256': sha256(args.converter_root / 'source.zip'),
                'files': {name: {'bytes': (args.output / name).stat().st_size, 'sha256': sha256(args.output / name)}
                          for name in (destination.name,) + SUPPORT_FILES},
                'quantized_evaluation': 'pending', 'uploaded': False}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'model': str(destination), **manifest['files'][destination.name]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
