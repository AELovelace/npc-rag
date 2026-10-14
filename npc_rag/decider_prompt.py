"""One token encoder for training and the experimental fixed-prefix inference layout."""
import json
from types import SimpleNamespace
from .decider_policy import DECIDER_OPTIONS, DECIDER_QUESTION, policy_for_layout


class FixedOrder:
    def shuffle(self, values):
        pass  # Inference uses GAME then CHAT; training supplies its seeded option-order RNG.


def checkpoint_layout(config):
    layout = config.get('routing_prompt_order', 'schema_first' if config.get('schema_first') else 'state_first')
    policy = policy_for_layout(layout)
    if config.get('routing_policy', policy) != policy:
        raise ValueError('Checkpoint routing policy disagrees with its prompt order')
    return layout


def build_decision_item(message, tokenizer, max_tokens, *, layout='state_first', rng=None, gold=0):
    from decider.prompt import build
    policy_for_layout(layout)  # Refuse unknown orders before constructing model inputs.
    state = json.dumps({'message': message}, ensure_ascii=False)
    if len(tokenizer.encode('Context:\n' + state, add_special_tokens=False)) > max_tokens:
        raise ValueError('Message exceeds token budget')
    example = SimpleNamespace(context=state, qs=[SimpleNamespace(
        text=DECIDER_QUESTION, options=list(DECIDER_OPTIONS), gold=gold)])
    item = build(example, tokenizer, rng=rng or FixedOrder(), max_options=2,
                 max_ctx_tokens=max_tokens, layout=layout)
    if len(item['ids']) > max_tokens or item['slots'] != [len(item['ids']) - 1]:
        raise ValueError('Full decision prompt exceeds budget or has unexpected slots')
    return item  # The answer slot contains no answer token; gold only selects the loss target.


def create_routing_decider(source, *, prompt_order='state_first', **kwargs):
    from decider.infer import Decider
    if prompt_order == 'state_first':
        return Decider(source, **kwargs)
    policy_for_layout(prompt_order)

    class SchemaFirstDecider(Decider):
        def _decide_items(self, requests, max_ctx_tokens=1536):
            items = []
            for state, questions in requests:
                if questions != [{'question': DECIDER_QUESTION, 'options': list(DECIDER_OPTIONS)}]:
                    raise ValueError('Schema-first routing accepts only the fixed routing schema')
                context = json.loads(state)
                if set(context) != {'message'}:
                    raise ValueError('Unexpected routing state fields')
                item = build_decision_item(context['message'], self.m.tok, max_ctx_tokens,
                                           layout='schema_first')
                item['types'] = ['choice']
                items.append(item)
            return requests, items  # Upstream scoring/temperature handling remains unchanged; GGUF still decodes in full.

    return SchemaFirstDecider(source, **kwargs)
