"""Validated per-request dialogue style; omitted styles preserve legacy persona behavior."""
import json
from typing import Literal

SpeakingStyle = Literal['normal', 'gentle', 'baby_talk']
SPEAKING_STYLES = ('normal', 'gentle', 'baby_talk')


def style_prompt(style: SpeakingStyle | None = None) -> str:
    if style is None:
        return ''  # Missing selectors leave existing personas and cached prefixes unchanged.
    if style not in SPEAKING_STYLES:
        raise ValueError('Unknown speaking_style; choose normal, gentle or baby_talk.')
    return '\n' + json.dumps({'speaking_style': style}, separators=(',', ':')) + '\n'
    # The fine-tuned student learns this compact control marker; no verbose style preset is injected.
