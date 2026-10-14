"""Optional installed-Decider encoder parity checks; no model downloads or GPU allocation."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(character) for character in text]


@unittest.skipUnless(importlib.util.find_spec('decider') and importlib.util.find_spec('torch'), 'Requires training environment')
class PromptLayoutContract(unittest.TestCase):
    def test_fixed_prefix_and_training_inference_parity(self):
        from npc_rag.decider_prompt import build_decision_item, create_routing_decider
        from npc_rag.decider_policy import DECIDER_QUESTION, DECIDER_OPTIONS
        from train_classifier import encode_rows
        tok = CharacterTokenizer()
        a = build_decision_item('first fixture', tok, 2048, layout='schema_first')
        b = build_decision_item('a different fixture', tok, 2048, layout='schema_first')
        prefix = a['prefix_len']
        self.assertEqual(a['ids'][:prefix], b['ids'][:prefix])
        self.assertTrue(''.join(map(chr, a['ids'])).startswith('Question: '))
        self.assertNotEqual(a['ids'][prefix:], b['ids'][prefix:])
        with patch('decider.infer.Decider.__init__', return_value=None):
            scorer = create_routing_decider('unused', prompt_order='schema_first')
        scorer.m = SimpleNamespace(tok=tok)
        requests = [(json.dumps({'message': 'first fixture'}), [{'question': DECIDER_QUESTION, 'options': list(DECIDER_OPTIONS)}])]
        _, inferred = scorer._decide_items(requests, 2048)
        trained = encode_rows([{'id': 'fixture', 'message': 'first fixture', 'label': 'general'}], tok, 2048, layout='schema_first')
        self.assertEqual(trained[0]['ids'], inferred[0]['ids'])
        self.assertEqual(trained[0]['slots'], inferred[0]['slots'])
        self.assertEqual(trained[0]['golds'], [1])

    def test_option_shuffle_preserves_gold_and_state_first_default(self):
        import random
        from npc_rag.decider_prompt import build_decision_item
        tok = CharacterTokenizer()
        default = build_decision_item('fixture', tok, 2048)
        self.assertTrue(''.join(map(chr, default['ids'])).startswith('Context:\n'))
        for seed in range(10):
            item = build_decision_item('fixture', tok, 2048, layout='schema_first', rng=random.Random(seed), gold=1)
            self.assertEqual(item['perms'][0][item['golds'][0]], 1)


if __name__ == '__main__':
    unittest.main()
