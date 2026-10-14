"""Final reports must not turn changed wiki evidence or a one-class collapse into an improvement."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from compare_classifier_runs import compare


def report():
    return {'policy': 'hybrid-v2', 'evidence_url': 'http://wiki', 'wiki_evidence': .66,
            'keyword_coverage': .5, 'keyword_dense': .62, 'raw_model': False, 'wiki_only': False,
            'classifier': {'decision_policy': 'game-chat-v3', 'engine': 'torch-bf16'},
            'datasets': {'held_out': {'accuracy': .9, 'total': 80}, 'mixed_intent': {'accuracy': 1.0, 'total': 16}},
            'results': [{'message': 'fixture', 'expected': 'game', 'dataset': 'held_out',
                         'method': 'llm', 'scores': {'wiki_best': .2}}],
            'summary': {'accuracy': .9, 'by_class': {'game': {'accuracy': .8}, 'general': {'accuracy': 1.0}},
                        'game_missed_as_chat': 2, 'chat_sent_to_game': 0}}


def test_matching_comparison_reports_both_error_directions():
    base, tuned = report(), report()
    tuned['summary']['accuracy'] = .95
    tuned['summary']['by_class']['game']['accuracy'] = .9
    tuned['summary']['game_missed_as_chat'] = 1
    result = compare(base, tuned)
    assert result['held_out_improved']
    assert result['fine_tuned']['game_missed_as_chat'] == 1
    assert result['fine_tuned']['chat_sent_to_game'] == 0


def test_changed_evidence_cannot_be_reported_as_a_model_gain():
    base = report()
    tuned = deepcopy(base)
    tuned['results'][0]['scores']['wiki_best'] = .7
    with pytest.raises(ValueError, match='Wiki evidence'):
        compare(base, tuned)


def test_same_score_is_not_an_improvement():
    assert not compare(report(), report())['held_out_improved']


def test_qwen_target_accepts_rounded_93_point_8_and_requires_perfect_mixed():
    base, tuned = report(), report()
    tuned['datasets']['held_out']['accuracy'] = 75 / 80
    assert compare(base, tuned)['qwen_target']['fine_tuned']['passed']
    tuned['datasets']['mixed_intent']['accuracy'] = 15 / 16
    assert not compare(base, tuned)['qwen_target']['fine_tuned']['passed']
