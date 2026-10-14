"""Convert a streaming token feed into ordered, speakable segments.

Rules (see design doc §6):
- prefer sentence-ending punctuation once the buffer has at least
  `min_words` words or `min_chars` characters;
- accept a comma/semicolon clause break once the buffer is long enough;
- force a break near `max_chars` so TTS never waits on a run-on sentence;
- never split abbreviations, decimals, URLs, or short quoted spans;
- segment IDs are ordered so audio always plays in response order.

Time-based flushing (`max_wait_ms`) is driven by the pipeline calling
`flush_stale()`; the segmenter itself holds no clocks or tasks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import ThinkTagStreamFilter, normalize_for_speech

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "a.m", "p.m", "no", "dept", "approx", "est", "inc", "ltd", "co", "corp",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun", "u.s", "u.k",
}

_SENTENCE_END = ".!?"
_CLAUSE_END = ",;:"


@dataclass
class Segment:
    segment_id: int
    text: str  # normalized, speakable
    raw: str  # original slice of the model output
    final: bool = False  # last segment of the generation


class SpeechSegmenter:
    def __init__(
        self,
        min_words: int = 8,
        min_chars: int = 40,
        max_chars: int = 140,
        first_min_words: int = 4,
        first_min_chars: int = 16,
        first_fast_open: bool = True,
    ) -> None:
        self._min_words = min_words
        self._min_chars = min_chars
        self._max_chars = max_chars
        # Emit a short complete opening sentence/clause ("Sure." /
        # "Sure,") immediately, below the length minimums. Disabled for
        # autoregressive voices that need full sentences for prosody.
        self._first_fast_open = first_fast_open
        # The first segment gates end-of-speech -> first audio, so it gets
        # lower thresholds and may break at a clause boundary; later
        # segments keep the longer rules for natural prosody.
        self._first_min_words = first_min_words
        self._first_min_chars = first_min_chars
        self._buffer = ""
        self._next_id = 0
        self._reasoning_filter = ThinkTagStreamFilter()

    # ---- boundary detection -----------------------------------------------

    def _is_safe_sentence_break(self, text: str, idx: int) -> bool:
        """Is text[idx] (a sentence-ending char) a real end of sentence?"""
        ch = text[idx]
        after = text[idx + 1] if idx + 1 < len(text) else ""
        if ch == "." and after.isdigit():
            return False  # decimal like 3.14
        if after and not after.isspace():
            return False  # mid-token: URLs, version numbers, file names
        if ch == ".":
            word = re.split(r"\s", text[:idx])[-1].lstrip("(\"'").lower()
            if word.rstrip(".") in _ABBREVIATIONS or word in _ABBREVIATIONS:
                return False
            if len(word) == 1 and word.isalpha():
                return False  # initials like "J."
        # Don't break inside an open double-quoted span.
        if text[: idx + 1].count('"') % 2 == 1:
            return False
        return True

    def _thresholds(self) -> tuple[int, int, bool]:
        """Progressive settle: (min_words, min_chars, clause_break_ok).

        The first segment accepts short clauses to start playback quickly.
        The second uses midpoint thresholds to reduce the gap after an
        opener. Later segments use full sentence rules for smoother speech.
        """
        if self._next_id == 0:
            return self._first_min_words, self._first_min_chars, True
        if self._next_id == 1:
            return (
                (self._first_min_words + self._min_words + 1) // 2,
                (self._first_min_chars + self._min_chars + 1) // 2,
                True,
            )
        return self._min_words, self._min_chars, False

    # Segment 0 gates first-audio latency AND its synthesis time: TTS cost
    # scales with segment length, so cap the opener hard (~80 chars is
    # ~1.5s of speech). Later segments keep the full max for prosody.
    _FIRST_MAX_CHARS = 80

    def _find_break(self, force: bool) -> int | None:
        """Return an index to cut after, or None to keep buffering."""
        buf = self._buffer
        min_words, min_chars, clause_ok = self._thresholds()
        if self._next_id == 0 and self._first_fast_open:
            # Release a short opener at a sentence or clause boundary;
            # models may use a comma even when the prompt asks for a period.
            for i, ch in enumerate(buf):
                if ch in _SENTENCE_END and i >= 2 and self._is_safe_sentence_break(buf, i):
                    return i
                if ch in _CLAUSE_END and i >= 4 and self._is_safe_sentence_break(buf, i):
                    return i
        max_chars = (
            min(self._max_chars, self._FIRST_MAX_CHARS)
            if self._next_id == 0
            else self._max_chars
        )
        long_enough = len(buf) >= min_chars or len(buf.split()) >= min_words
        if long_enough:
            for i, ch in enumerate(buf):
                if ch in _SENTENCE_END and i + 1 >= min_chars // 2:
                    if self._is_safe_sentence_break(buf, i):
                        return i
            if clause_ok:
                # Fast first phrase: a clause boundary is good enough to
                # start speaking; the rest of the sentence follows next.
                for i, ch in enumerate(buf):
                    if ch in _CLAUSE_END and i + 1 >= min_chars:
                        if self._is_safe_sentence_break(buf, i):
                            return i
        if len(buf) >= max_chars:
            # Overlong: take the last clause break, else the last space.
            window = buf[:max_chars]
            for punct in _CLAUSE_END:
                idx = window.rfind(punct)
                if idx > self._min_chars // 2 and self._is_safe_sentence_break(window, idx):
                    return idx
            idx = window.rfind(" ")
            if idx > 0:
                return idx
            return len(window) - 1
        if force and long_enough:
            for punct in _CLAUSE_END:
                idx = buf.rfind(punct)
                if idx > self._min_chars // 2:
                    return idx
        return None

    def _emit(self, cut_after: int) -> Segment | None:
        raw = self._buffer[: cut_after + 1]
        self._buffer = self._buffer[cut_after + 1 :].lstrip()
        text = normalize_for_speech(raw)
        if not text:
            return None
        seg = Segment(segment_id=self._next_id, text=text, raw=raw)
        self._next_id += 1
        return seg

    # ---- public API --------------------------------------------------------

    def feed(self, delta: str) -> list[Segment]:
        """Add a token delta; return zero or more completed segments."""
        clean = self._reasoning_filter.feed(delta)
        if not clean:
            return []
        self._buffer += clean
        out: list[Segment] = []
        while (cut := self._find_break(force=False)) is not None:
            if (seg := self._emit(cut)) is not None:
                out.append(seg)
        return out

    def flush_stale(self) -> list[Segment]:
        """Called by the pipeline when max_wait elapsed with audio idle."""
        out: list[Segment] = []
        if (cut := self._find_break(force=True)) is not None:
            if (seg := self._emit(cut)) is not None:
                out.append(seg)
        return out

    def finish(self) -> list[Segment]:
        """Generation complete: flush whatever remains."""
        out: list[Segment] = []
        while (cut := self._find_break(force=False)) is not None:
            if (seg := self._emit(cut)) is not None:
                out.append(seg)
        if self._buffer.strip():
            if (seg := self._emit(len(self._buffer) - 1)) is not None:
                out.append(seg)
        self._buffer = ""
        if out:
            out[-1].final = True
        return out
