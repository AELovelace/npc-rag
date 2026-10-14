"""Package the evaluated participation release and pin assets for opt-in startup downloads."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

from export_classifier_gguf import ROOT, SUPPORT_FILES, sha256
sys.path.insert(0, str(ROOT))
from npc_rag.model_download import CONFIG_FILES
from npc_rag.participation_policy import validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=ROOT / 'runs/releases/decider-chat-0-1-q4km')
    args = parser.parse_args()
    release, name = args.release, 'decider-chat-0-1'
    validate_checkpoint(json.loads((release / 'decider_config.json').read_text()),
                        json.loads((release / 'routing_policy.json').read_text()))
    report = json.loads((release / 'quantized-participation.json').read_text())
    for split in ('development', 'test'):
        rows = report['datasets'][split]['rows']
        if len(rows) != 32 or len({r['id'] for r in rows}) != 32 or any(r['predicted'] not in ('join', 'quiet') for r in rows):
            raise ValueError('A complete 32-case native evaluation is required for each split')
    qualification = {'task': 'participation', 'format': 'Q4_K_M', 'policy': report['model']['decision_policy'],
                     'datasets': {s: {k: v for k, v in r.items() if k != 'rows'} for s, r in report['datasets'].items()},
                     'evaluation_sha256': sha256(release / 'quantized-participation.json'),
                     'scope': 'Small synthetic pilot; neither live Qwen A/B nor production behavior qualification',
                     'calibration': report['calibration'], 'deployment_ready': False, 'uploaded': False}
    (release / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n', encoding='utf-8')
    shutil.copy2(ROOT / 'runs/gguf-export-tools/Apache-2.0.txt', release / 'LICENSE.txt')
    (release / 'NOTICE.txt').write_text(
        'LiDollQuest Decider Chat 0.1\nDerived from Mapika/decider-2b (Apache-2.0), revision '
        'ff99089b6810691bd79134ec6febb5d9afa6e619.\n'
        'Changes: rank-8 LoRA sociable JOIN/QUIET training, epoch 2, merged into BF16, '
        'schema-first prompt, Q4_K_M quantization. Original publisher does not endorse this adaptation.\n', encoding='utf-8')
    (release / 'HOSTING.txt').write_text(
        'Upload both files:\nhttps://lidoll.dev/gguf/decider-chat-0-1.gguf\n'
        'https://lidoll.dev/gguf/decider-chat-0-1-support.zip\n\n'
        'Pull the prepared npc-rag code and set PARTICIPATION_ENABLED=true in each RAG .env; restart.\n'
        'Uses existing DECIDER_GPU_LAYERS/DEVICES/THREADS and port 9092; adds another model per GPU.\n'
        'Startup checks hashes, downloads missing assets, and takes configs from npc_rag/participation_config in Git.\n'
        'Check /health participation.enabled, healthy and workers before B.\n'
        'Game-server switch: QUEST_PARTICIPATION_BACKEND=decider, then restart updated game server.\n'
        'Rollback game gate with QUEST_PARTICIPATION_BACKEND=qwen; disable RAG participation to release its VRAM.\n'
        'This is JOIN/QUIET, not GAME/CHAT. Do not replace decider0-1.gguf or its configs.\n'
        'The release remains an experimental pilot; see qualification.json.\n', encoding='utf-8')
    additional = ('qualification.json', 'LICENSE.txt', 'NOTICE.txt', 'HOSTING.txt')
    manifest = json.loads((release / 'manifest.json').read_text())
    manifest['quantized_evaluation'] = 'qualification.json'
    manifest['files'] = {file: {'bytes': (release / file).stat().st_size, 'sha256': sha256(release / file)}
                         for file in (name + '.gguf',) + SUPPORT_FILES + additional}
    (release / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    archive = release / (name + '-support.zip')
    support = SUPPORT_FILES + additional + ('manifest.json',)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for file in support:
            bundle.write(release / file, arcname=file)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError('Support archive CRC verification failed')
    pin = {'name': name, 'download_url': manifest['download_url'],
           'support_url': 'https://lidoll.dev/gguf/' + archive.name,
           'files': {file: {'bytes': (release / file).stat().st_size, 'sha256': sha256(release / file)}
                     for file in (name + '.gguf',) + support},
           'support_archive': {'bytes': archive.stat().st_size, 'sha256': sha256(archive)}}
    (ROOT / 'npc_rag/participation_release.json').write_text(json.dumps(pin, indent=2) + '\n', encoding='utf-8')
    configs = ROOT / 'npc_rag/participation_config'
    configs.mkdir(exist_ok=True)
    for file in CONFIG_FILES:
        shutil.copy2(release / file, configs / file)  # Runtime uses these Git-owned files instead of the ZIP's copies.
    (release / 'SHA256SUMS.txt').write_text(''.join(f'{sha256(file)}  {file.name}\n' for file in (release / (name + '.gguf'), archive)), encoding='ascii')
    print(json.dumps({'model': str(release / (name + '.gguf')), 'support': str(archive),
                      'model_spec': pin['files'][name + '.gguf'], 'support_spec': pin['support_archive']}, indent=2))


if __name__ == '__main__':
    main()
