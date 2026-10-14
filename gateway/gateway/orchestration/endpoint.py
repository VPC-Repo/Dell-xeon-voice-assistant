"""Semantic turn-completion heuristic.

At the speculative-silence point we have a transcript of what was said so
far. If it reads as a complete utterance, waiting the rest of the long
end-silence window adds pure latency; if it ends mid-thought ("...and
the", "...because"), committing early would cut the user off. This
heuristic gates a shorter end-silence for turns that look finished.

Works on punctuated and unpunctuated
transcripts. Deliberately conservative: unknown shapes count as
incomplete, because a false cutoff feels worse than 200 ms of patience.
"""

from __future__ import annotations

import re

# Trailing words that almost always continue: conjunctions, prepositions,
# articles, auxiliaries, fillers, possessives.
_INCOMPLETE_ENDINGS = {
    "and", "or", "but", "so", "because", "although", "though", "if", "when",
    "while", "unless", "until", "than", "then",
    "the", "a", "an", "to", "of", "in", "on", "for", "with", "at", "by",
    "from", "about", "into", "over", "under", "between", "after", "before",
    "that", "which", "who", "whose", "whom", "where", "how", "why", "what",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "my", "your", "his", "her", "its", "our", "their",
    "um", "uh", "er", "uhh", "umm", "hmm", "like", "very", "really", "quite",
    "please", "i",
}

_WORD = re.compile(r"[a-z']+")


def looks_complete(text: str) -> bool:
    """Does this transcript read as a finished utterance?"""
    t = text.strip()
    if not t:
        return False
    words = t.split()
    if len(words) < 2:
        return False  # one word is ambiguous; wait for the full window
    if t.endswith((".", "!", "?")):
        return True
    if t.endswith((",", ":", ";", "-", "—")):
        return False
    last = _WORD.findall(words[-1].lower())
    if not last:
        return False
    return last[-1] not in _INCOMPLETE_ENDINGS
