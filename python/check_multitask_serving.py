"""Replay the held-out decisions through the real shared serving interfaces, without deployment."""
import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npc_rag.config import load_settings
from npc_rag.classifier_backend import make_classifier
from npc_rag.participation import make_participation


async def check(release):
    manifest = json.loads((release / 'manifest.json').read_text(encoding='utf-8'))
    settings = replace(load_settings(str(release / 'unused.env')), classifier_backend='decider', decider_model=str(release.resolve()),
                       decider_gguf_file=manifest['name'] + '.gguf', decider_gpu_layers=0, participation_enabled=True, participation_model='shared')
    routing = make_classifier(settings)
    participation = make_participation(settings, classifier=routing)
    report = json.loads((release / 'quantized-multitask.json').read_text(encoding='utf-8'))
    inputs = {}
    for task, source in report['evaluation_sources'].items():
        for line in Path(source['dataset']).read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            inputs[(task, row['id'])] = row
    results = []
    try:
        await routing.start()
        await participation.start()
        for reference in report['datasets']['synthetic_test']['rows']:
            task = reference['task']
            row = inputs[(task, reference['id'])]
            if task == 'routing':
                decision = await routing.decide(row['message'])
                label, probability = str(decision), decision.game_probability
            else:
                decision = await participation.decide(row['state'])
                label, probability = decision
            results.append({'id': row['id'], 'task': task, 'predicted': label, 'expected': reference['expected'],
                            'matches_offline': label == reference['predicted'], 'first_option_probability': probability})
            if len(results) % 16 == 0:
                print(f"Checked {len(results)}/{len(report['datasets']['synthetic_test']['rows'])} serving decisions", flush=True)
        output = {'loaded_model_instances': 1, 'shared_backend': routing.backend is participation.backend,
                  'all_labels_match_offline': all(row['matches_offline'] for row in results),
                  'datasets': {task: {'correct': sum(row['predicted'] == row['expected'] for row in results if row['task'] == task),
                                      'total': sum(row['task'] == task for row in results)} for task in ('routing', 'participation')},
                  'rows': results}
        (release / 'serving-check.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({key: value for key, value in output.items() if key != 'rows'}, indent=2))
        if not output['all_labels_match_offline']:
            raise RuntimeError('Serving decisions differ from offline evaluation')
    finally:
        await participation.aclose()
        await routing.aclose()  # Drain uncancellable work and release the single engine even on failure.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=ROOT / 'runs/releases/decider-multitask-v3-q4km')
    asyncio.run(check(parser.parse_args().release))
