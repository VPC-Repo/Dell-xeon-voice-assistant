"""Streaming linear resampler for mono s16le PCM.

Backends emit at their native rate (Kokoro is 24000 Hz);
the gateway asks for TTS_SAMPLE_RATE (24000 by default). Linear
interpolation is audibly fine for speech and keeps this dependency-light.
State (fractional read position + last sample) carries across chunks so
chunk boundaries stay continuous.
"""

from __future__ import annotations

import numpy as np


class StreamResampler:
    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._ratio = src_rate / dst_rate
        self._pos = 0.0  # fractional index into the source stream
        self._carry = np.zeros(0, dtype=np.int16)

    def push(self, pcm: bytes) -> bytes:
        if self.src_rate == self.dst_rate:
            return pcm
        samples = np.frombuffer(pcm, dtype=np.int16)
        buf = np.concatenate([self._carry, samples])
        if len(buf) < 2:
            self._carry = buf
            return b""
        # Output indices whose interpolation window lies inside buf.
        out_positions = np.arange(self._pos, len(buf) - 1, self._ratio)
        if len(out_positions) == 0:
            self._carry = buf
            return b""
        idx = out_positions.astype(np.int64)
        frac = out_positions - idx
        out = ((1 - frac) * buf[idx] + frac * buf[idx + 1]).astype(np.int16)
        # Keep the tail needed by the next chunk's interpolation.
        next_pos = out_positions[-1] + self._ratio
        keep_from = int(next_pos)
        self._pos = next_pos - keep_from
        self._carry = buf[keep_from:]
        return out.tobytes()

    def flush(self) -> bytes:
        """Emit the final carried sample(s) as-is."""
        out = self._carry.tobytes()
        self._carry = np.zeros(0, dtype=np.int16)
        self._pos = 0.0
        return out if self.src_rate != self.dst_rate else b""
