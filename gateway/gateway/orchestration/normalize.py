"""Normalize LLM text into speakable form before synthesis.

The system prompt asks Qwen for clean spoken text, but the orchestration
layer must not trust that: Markdown, reasoning tags, URLs, and symbols are
stripped here regardless of what the model was told.
"""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_RAW_URL = re.compile(r"https?://[^\s)>\]]+")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_MD_NUMBULLET = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_CITATION = re.compile(r"\[\d+(?:,\s*\d+)*\]")
# Stage directions the model may emit despite instructions: [pause],
# [sigh], [laughs softly], ... Words in brackets, never spoken.
_STAGE_TAG = re.compile(r"\[\s*[A-Za-z][A-Za-z _'-]{1,24}\]")
_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "️"
    "]+"
)
_REPEAT_PUNCT = re.compile(r"([!?.,;:])\1+")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{2,}")


def normalize_for_speech(text: str) -> str:
    """Convert one already-segmented phrase of model text to plain speech."""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_OPEN.sub("", text)  # unterminated reasoning must never be spoken
    text = _CODE_FENCE.sub(" (code omitted) ", text)
    text = _MD_TABLE_ROW.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _RAW_URL.sub("a link", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BULLET.sub("", text)
    text = _MD_NUMBULLET.sub("", text)
    # Run emphasis twice to unwrap nested markers like ***bold italic***.
    text = _MD_EMPHASIS.sub(r"\2", text)
    text = _MD_EMPHASIS.sub(r"\2", text)
    text = _CITATION.sub("", text)
    text = _STAGE_TAG.sub(" ", text)
    text = _EMOJI.sub("", text)
    text = _REPEAT_PUNCT.sub(r"\1", text)
    text = text.replace("—", ", ").replace("–", ", ")
    text = _MULTI_NEWLINE.sub(". ", text)
    text = text.replace("\n", " ")
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def strip_reasoning(text: str) -> str:
    """Remove complete or dangling <think> blocks from display text."""
    return _THINK_OPEN.sub("", _THINK_BLOCK.sub("", text)).strip()


def strip_stage_tags(text: str) -> str:
    """Remove bracketed stage cues from a complete text."""
    return _MULTI_SPACE.sub(" ", _STAGE_TAG.sub(" ", text)).strip()


class StageTagStreamFilter:
    """Drop bracketed stage cues ([pause], [sigh]) from a token stream,
    handling tags split across deltas. Non-tag bracket content (numbers,
    long spans) passes through untouched."""

    _MAX_TAG = 26

    def __init__(self) -> None:
        self._hold = ""

    def feed(self, delta: str) -> str:
        text = self._hold + delta
        self._hold = ""
        out: list[str] = []
        while text:
            i = text.find("[")
            if i == -1:
                out.append(text)
                break
            out.append(text[:i])
            rest = text[i:]
            j = rest.find("]")
            if j == -1:
                if len(rest) <= self._MAX_TAG:
                    self._hold = rest  # possible tag still streaming in
                else:
                    out.append(rest)  # too long to be a stage cue
                break
            candidate = rest[: j + 1]
            if not _STAGE_TAG.fullmatch(candidate):
                out.append(candidate)
            text = rest[j + 1 :]
        return "".join(out)

    def flush(self) -> str:
        held, self._hold = self._hold, ""
        return held


class ThinkTagStreamFilter:
    """Drop <think>...</think> spans from a token stream, incrementally.

    Whitespace-preserving (unlike strip_reasoning, which is for complete
    texts): deltas pass through byte-for-byte outside think spans, so
    joining the outputs reconstructs the exact display text. Handles tags
    split across deltas by holding back possible tag prefixes.
    """

    def __init__(self) -> None:
        self._in_think = False
        self._holdback = ""

    def feed(self, delta: str) -> str:
        text = self._holdback + delta
        self._holdback = ""
        out = []
        i = 0
        while i < len(text):
            tag = "</think>" if self._in_think else "<think>"
            idx = text.find(tag, i)
            if idx == -1:
                remainder = text[i:]
                for k in range(min(len(tag) - 1, len(remainder)), 0, -1):
                    if tag.startswith(remainder[-k:]):
                        self._holdback = remainder[-k:]
                        remainder = remainder[:-k]
                        break
                if not self._in_think:
                    out.append(remainder)
                break
            if not self._in_think:
                out.append(text[i:idx])
            self._in_think = not self._in_think
            i = idx + len(tag)
        return "".join(out)


# STT non-speech annotations: recognizers emit markers like [BLANK_AUDIO],
# [ Silence ], (wind blowing), *coughs*, ♪ for audio without speech. If a
# transcript is only annotations, no one actually said anything -- treating
# it as user input makes the assistant chat with room noise in a loop.
_STT_ANNOTATION = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|[♪♫]+")


def strip_stt_annotations(text: str) -> str:
    """Remove non-speech markers from an STT transcript."""
    cleaned = _MULTI_SPACE.sub(" ", _STT_ANNOTATION.sub(" ", text.replace("\n", " ")))
    cleaned = cleaned.strip()
    # Leftover punctuation with no words means nothing was actually said.
    if not re.search(r"[A-Za-z0-9]", cleaned):
        return ""
    return cleaned
