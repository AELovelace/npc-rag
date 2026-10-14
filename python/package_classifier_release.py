"""Package a locally evaluated GGUF and its matching support files for operator hosting."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

from export_classifier_gguf import ROOT, SUPPORT_FILES, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=ROOT / 'runs/releases/decider0-1-q4km')
    parser.add_argument('--bf16-report', type=Path, default=ROOT / 'runs/classifier-v3-first-pass/tuned-heldout.json')
    args = parser.parse_args()
    release = args.release
    quant = json.loads((release / 'quantized-heldout.json').read_text(encoding='utf-8'))
    reference = json.loads(args.bf16_report.read_text(encoding='utf-8'))
    if len(quant['results']) != 96 or quant['classifier']['failures'] or any(r['method'] == 'error' for r in quant['results']):
        raise ValueError('A complete error-free 96-case evaluation is required')
    same_evidence = len(reference['results']) == len(quant['results']) and all(
        all(a[k] == b[k] for k in ('message', 'expected', 'dataset')) and
        all(a.get('scores', {}).get(k) == b.get('scores', {}).get(k) for k in ('wiki_best', 'wiki_keyword'))
        for a, b in zip(reference['results'], quant['results']))
    heldout, mixed = quant['datasets']['held_out'], quant['datasets']['mixed_intent']
    passed = heldout['total'] == 80 and mixed['total'] == 16 and heldout['accuracy'] >= .9375 and mixed['accuracy'] == 1
    qualification = {'format': 'Q4_K_M', 'classifier': quant['classifier'], 'summary': quant['summary'],
                     'held_out': heldout, 'mixed': mixed, 'requested_accuracy_target_passed': passed,
                     'wiki_evidence_matches_bf16': same_evidence, 'bf16_summary': reference['summary'],
                     'raw_evaluation_sha256': sha256(release / 'quantized-heldout.json'),
                     'latency_scope': 'Sequential local CPU evaluation only; no server load-test or median reply-time claim',
                     'calibration': 'temperature 1; not refitted after training or quantization', 'deployed': False}
    (release / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n', encoding='utf-8')
    shutil.copy2(ROOT / 'runs/gguf-export-tools/Apache-2.0.txt', release / 'LICENSE.txt')
    (release / 'NOTICE.txt').write_text(
        'LiDollQuest Decider 0.1\nDerived from Mapika/decider-2b (Apache-2.0), revision '
        'ff99089b6810691bd79134ec6febb5d9afa6e619.\n'
        'Local modifications: rank-8 LoRA domain training, selected epoch 2, merged into BF16, '
        'converted and quantized to Q4_K_M. The original publisher does not endorse this adaptation.\n', encoding='utf-8')
    (release / 'HOSTING.txt').write_text(
        'LiDollQuest Decider 0.1 - Q4_K_M\n\n'
        'Upload these TWO files beside each other:\n'
        '  https://lidoll.dev/gguf/decider0-1.gguf\n'
        '  https://lidoll.dev/gguf/decider0-1-support.zip\n\n'
        'On each updated npc-rag server, extract the support zip and place the GGUF in the same local folder.\n'
        'The native Decider adapter needs the matching tokenizer and decider_config.json; the GGUF alone is insufficient.\n'
        'Use a LOCAL folder for DECIDER_MODEL, not the download URL. For example:\n'
        'CLASSIFIER_BACKEND=decider\nDECIDER_MODEL=data/models/decider0-1\n'
        'DECIDER_GGUF_FILE=decider0-1.gguf\nDECIDER_THREADS=4\nDECIDER_GPU_LAYERS=0\n\n'
        'This is game-chat-v3, message-first. It is NOT the separate prefix-first smoke-test adapter.\n'
        'Update npc-rag to code supporting that policy before loading. Verify SHA256 against manifest.json.\n'
        'See qualification.json for native Q4_K_M accuracy. No service configuration was changed by packaging.\n', encoding='utf-8')
    manifest_path = release / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    additional = ('qualification.json', 'LICENSE.txt', 'NOTICE.txt', 'HOSTING.txt')
    for name in additional:
        manifest['files'][name] = {'bytes': (release / name).stat().st_size, 'sha256': sha256(release / name)}
    manifest['quantized_evaluation'] = 'qualification.json'
    manifest['accuracy_qualified'] = passed
    manifest['support_url'] = 'https://lidoll.dev/gguf/decider0-1-support.zip'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    archive = release / 'decider0-1-support.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name in SUPPORT_FILES + additional + ('manifest.json',):
            bundle.write(release / name, arcname=name)  # Never include source weights, intermediate GGUFs, logs or secrets.
    files = (release / 'decider0-1.gguf', archive)
    (release / 'SHA256SUMS.txt').write_text(''.join(f'{sha256(path)}  {path.name}\n' for path in files), encoding='ascii')
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError('Support archive CRC validation failed')
    print(json.dumps({'accuracy_qualified': passed, 'wiki_evidence_matches_bf16': same_evidence,
                      'files': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in files]}, indent=2))


if __name__ == '__main__':
    main()
