"""Package the evaluated combined Decider and pin its upload assets for serving."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

from export_classifier_gguf import ROOT, SUPPORT_FILES, sha256
sys.path.insert(0, str(ROOT))
from npc_rag.model_download import CONFIG_FILES
from npc_rag.multitask import GGUF, MODEL, validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=ROOT / 'runs/releases/decider-multitask-v3-q4km')
    parser.add_argument('--accept-known-opening-miss', action='store_true')
    args = parser.parse_args()
    release, name = args.release, MODEL.split('/')[-1]
    validate_checkpoint(json.loads((release / 'decider_config.json').read_text(encoding='utf-8')),
                        json.loads((release / 'routing_policy.json').read_text(encoding='utf-8')))
    manifest = json.loads((release / 'manifest.json').read_text(encoding='utf-8'))
    for file, spec in manifest['files'].items():
        if (release / file).stat().st_size != spec['bytes'] or sha256(release / file) != spec['sha256']:
            raise ValueError('Export asset changed: ' + file)
    report = json.loads((release / 'quantized-multitask.json').read_text(encoding='utf-8'))
    rows = report['datasets']['synthetic_test']['rows']
    for task, total in (('routing', 100), ('participation', 176)):
        selected = [row for row in rows if row['task'] == task]
        if len(selected) != total or len({row['id'] for row in selected}) != total:
            raise ValueError('A complete v3 native evaluation is required')
    targeted_path = ROOT / 'runs/multitask-v3-hired-opening-20261015/targeted-comparison.json'
    targeted = json.loads(targeted_path.read_text(encoding='utf-8'))
    failed = [key for key, passed in targeted['gates'].items() if not passed]
    if failed != ['new_hired_openings_all_join'] or not args.accept_known_opening_miss:
        raise ValueError('This release requires the documented user acceptance of its single opening gate failure')
    shutil.copy2(targeted_path, release / 'targeted-comparison.json')
    qualification = {key: value for key, value in report.items() if key != 'datasets'}
    qualification['release_decision'] = {'user_authorized': True, 'known_failed_gates': failed,
        'reason': 'User accepted the known miss and chose rollout with v2 available for rollback.',
        'live_shared_pool_load_tested': False, 'deployed': False}

    qualification['datasets'] = {split: {key: value for key, value in result.items() if key != 'rows'}
                                 for split, result in report['datasets'].items()}
    qualification['evaluation_sha256'] = sha256(release / 'quantized-multitask.json')
    qualification['model_sha256'] = sha256(release / GGUF)
    (release / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n', encoding='utf-8')
    shutil.copy2(ROOT / 'runs/gguf-export-tools/Apache-2.0.txt', release / 'LICENSE.txt')
    (release / 'NOTICE.txt').write_text(
        'LiDollQuest Decider Multitask v3\nDerived from Mapika/decider-2b (Apache-2.0), revision '
        'ff99089b6810691bd79134ec6febb5d9afa6e619.\n'
        'Changes: rank-8 QLoRA on GAME/CHAT and JOIN/QUIET, selected epoch 1 of 2, merged into BF16, '
        'per-task prompts, Q4_K_M quantization. Original publisher does not endorse this adaptation.\n', encoding='utf-8')
    (release / 'HOSTING.txt').write_text(
        'Upload both files to https://lidoll.dev/gguf/:\n' + GGUF + '\n' + name + '-support.zip\n\n'
        'First deploy the updated Lidollquest-server participation validator (accepts old and new releases).\n'
        'Install the updated npc-rag code on each RAG host, then set these .env values:\n'
        'CLASSIFIER_BACKEND=decider\nDECIDER_MODEL=' + MODEL + '\nDECIDER_GGUF_FILE=' + GGUF + '\n'
        'PARTICIPATION_ENABLED=true\nPARTICIPATION_MODEL=shared\nPARTICIPATION_TIMEOUT=10\n\n'
        'Keep existing DECIDER_GPU_LAYERS/DEVICES/THREADS settings. Restart RAG.\n'
        'The managed installer verifies/downloads one model; both tasks share one worker per GPU.\n'
        'Check /health classification and participation: shared_pool=true, matching worker PIDs, healthy workers.\n'
        'Keep the game server on QUEST_PARTICIPATION_BACKEND=decider. No new port is needed.\n'
        'Manual offline installation: extract this ZIP and put the GGUF into data/models/decider-multitask-v3.\n'
        'Known limitation: 7/8 new hired openings pass; the keepsake-box opening still gets QUIET.\n'
        'The user accepted this miss for rollout; offline gate results remain unchanged.\n'
        'Rollback: DECIDER_MODEL=lidoll/decider-multitask-v2, DECIDER_GGUF_FILE=decider-multitask-v2.gguf, '
        'PARTICIPATION_MODEL=shared; restart.\n'
        'Synthetic/offline scores are recorded in qualification.json; confidence is uncalibrated.\n', encoding='utf-8')
    additional = ('qualification.json', 'targeted-comparison.json', 'LICENSE.txt', 'NOTICE.txt', 'HOSTING.txt')
    spec = lambda file: {'bytes': (release / file).stat().st_size, 'sha256': sha256(release / file)}
    manifest['quantized_evaluation'] = 'qualification.json'
    manifest['files'] = {file: spec(file) for file in (GGUF,) + SUPPORT_FILES + additional}
    (release / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    support = SUPPORT_FILES + additional + ('manifest.json',)
    archive = release / (name + '-support.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for file in support:
            bundle.write(release / file, arcname=file)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError('Support archive CRC verification failed')
    pin = {'name': name, 'download_url': manifest['download_url'],
           'support_url': 'https://lidoll.dev/gguf/' + archive.name,
           'files': {file: spec(file) for file in (GGUF,) + support}, 'support_archive': spec(archive.name)}
    (ROOT / 'npc_rag/multitask_release.json').write_text(json.dumps(pin, indent=2) + '\n', encoding='utf-8')
    configs = ROOT / 'npc_rag/multitask_config'
    configs.mkdir(exist_ok=True)
    for file in CONFIG_FILES:
        shutil.copy2(release / file, configs / file)  # Copy the exact tested prompts into the distributable runtime.
    (release / 'SHA256SUMS.txt').write_text(''.join(f'{sha256(file)}  {file.name}\n' for file in (release / GGUF, archive)), encoding='ascii')
    print(json.dumps({'model': str(release / GGUF), 'support': str(archive), 'support_spec': pin['support_archive']}, indent=2))


if __name__ == '__main__':
    main()
