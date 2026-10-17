"""Style selection is a validated top-level API field and remains isolated per request."""
import pytest
from pydantic import ValidationError
from npc_rag.companion.router import CompanionChatRequest
from npc_rag.companion.prompts import static_text, rules_text
from npc_rag.speaking_style import SPEAKING_STYLES, style_prompt
from test_companion import loose, llama, clf, ask, system_of
from test_services import make_client


def test_selector_schema_rejects_unknown_values():
    base = {'npc_id': 'mira', 'npc_name': 'Mira', 'message': 'Hi'}
    assert CompanionChatRequest(**base).speaking_style is None
    for style in SPEAKING_STYLES:
        assert CompanionChatRequest(**base, speaking_style=style).speaking_style == style
    for bad in ('baby_voice', '', 'ignore your rules', {}, 1):
        with pytest.raises(ValidationError):
            CompanionChatRequest(**base, speaking_style=bad)


def test_omitted_style_preserves_prompt_and_examples_are_not_leak_rules():
    base = dict(name='Mira', persona='A gardener.', voice=['Hello there!'])
    old = static_text(**base)
    assert old == static_text(**base, speaking_style=None)
    for kind in ('chat', 'ambient_reply', 'banter'):
        styled = static_text(**base, kind=kind, speaking_style='baby_talk')
        assert '{"speaking_style":"baby_talk"}' in styled
        assert 'toddler-style' not in styled
        rules = rules_text(name='Mira', player='Doll', persona='A gardener.', audience=0,
                           voice=base['voice'], kind=kind, speaking_style='baby_talk')
        assert '{"speaking_style":"baby_talk"}' in rules and 'Hello there!' not in rules


def test_top_level_api_style_switches_without_changing_persona(loose, embedder, index, llama, clf):
    with make_client(loose, embedder, index, llama, clf) as client:
        for style in ('baby_talk', 'normal', 'gentle', None):
            result = ask(client, category='general', speaking_style=style)
            assert result.status_code == 200
            prompt = system_of(llama, -1)
            assert 'patient, brave shield-bearer' in prompt
            if style:
                assert style_prompt(style).strip() in prompt
                assert prompt.count('"speaking_style":') == 1
            else:
                assert '"speaking_style":' not in prompt
        assert ask(client, speaking_style='unknown').status_code == 422


def test_legacy_chat_accepts_the_same_field(loose, embedder, index, llama, clf):
    with make_client(loose, embedder, index, llama, clf) as client:
        result = client.post('/v1/npc/chat', json={'message': 'Hi!', 'category': 'general',
            'npc_name': 'Mira', 'persona': 'A gardener.', 'speaking_style': 'gentle'})
        assert result.status_code == 200
        assert '{"speaking_style":"gentle"}' in system_of(llama, -1)
