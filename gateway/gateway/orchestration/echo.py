"""Self-echo detection.

On speakerphones (laptops, phones held in hand) browser AEC is imperfect:
the assistant's own playback leaks into the microphone, gets transcribed,
and would start a new turn answering itself. This module decides whether a
transcript is probably the assistant's own recent speech.

Two signals, either sufficient:
- token coverage: nearly every word of the transcript appears in what the
  assistant spoke within the window (STT of an echo is a lossy subset);
- sequence similarity against any single recent segment.

Short transcripts (< 3 tokens) are too ambiguous for coverage ("yes",
"okay" legitimately reuse assistant words), so they require an exact
phrase match instead.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def is_probable_echo(
    transcript: str,
    recent_speech: list[str],
    threshold: float = 0.72,
) -> bool:
    if threshold <= 0 or not recent_speech:
        return False
    t_tokens = _tokens(transcript)
    if not t_tokens:
        return False

    if len(t_tokens) < 3:
        phrase = " ".join(t_tokens)
        return any(phrase in " ".join(_tokens(r)) for r in recent_speech)

    spoken_bag: dict[str, int] = {}
    for text in recent_speech:
        for tok in _tokens(text):
            spoken_bag[tok] = spoken_bag.get(tok, 0) + 1
    covered = 0
    for tok in t_tokens:
        if spoken_bag.get(tok, 0) > 0:
            spoken_bag[tok] -= 1
            covered += 1
    if covered / len(t_tokens) >= threshold:
        return True

    joined_t = " ".join(t_tokens)
    return any(
        SequenceMatcher(None, joined_t, " ".join(_tokens(r))).ratio() >= threshold
        for r in recent_speech
    )
