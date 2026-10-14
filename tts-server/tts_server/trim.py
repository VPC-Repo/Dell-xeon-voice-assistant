"""Streaming silence trimming for synthesized segments.

TTS engines pad output with variable lead-in and trail-out silence,
typically a couple of hundred milliseconds per segment. Stitched
sentence by sentence, those variable gaps read as jerky rhythm. This trims both ends and lets
the server append one *constant* tail instead, so inter-segment pauses are
uniform.

Works streaming: leading silence is skipped until the first audible block;
trailing silence is held back and only released if more audible audio
follows.
"""

from __future__ import annotations

import numpy as np

_BLOCK_MS = 10


class SilenceTrimmer:
    def __init__(self, sample_rate: int, threshold: int = 300) -> None:
        self._block_bytes = sample_rate * 2 * _BLOCK_MS // 1000
        self._threshold = threshold
        self._buffer = bytearray()
        self._lead_done = False
        self._held_silence = bytearray()

    def _audible(self, block: bytes) -> bool:
        samples = np.frombuffer(block, dtype=np.int16)
        return len(samples) > 0 and int(np.abs(samples).max()) >= self._threshold

    def push(self, pcm: bytes) -> bytes:
        self._buffer.extend(pcm)
        out = bytearray()
        while len(self._buffer) >= self._block_bytes:
            block = bytes(self._buffer[: self._block_bytes])
            del self._buffer[: self._block_bytes]
            if self._audible(block):
                if not self._lead_done:
                    self._lead_done = True
                if self._held_silence:
                    out += self._held_silence  # interior pause: keep it
                    self._held_silence.clear()
                out += block
            elif self._lead_done:
                self._held_silence += block  # maybe trailing: hold back
            # else: leading silence, drop
        return bytes(out)

    def flush(self) -> bytes:
        """End of stream: the remainder counts as audio if audible,
        held-back silence is dropped (that was the trailing pad)."""
        tail = bytes(self._buffer)
        self._buffer.clear()
        self._held_silence.clear()
        if tail and self._lead_done and self._audible(tail):
            return tail
        return b""
