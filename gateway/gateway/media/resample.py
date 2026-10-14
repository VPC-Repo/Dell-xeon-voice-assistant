"""PCM resampling helpers built on PyAV's libswresample bindings.

Two directions:
- inbound: browser Opus decoded by aiortc (48 kHz, often stereo) -> mono
  s16le at the STT rate (16 kHz);
- outbound: TTS PCM (24 kHz mono) -> 48 kHz mono for Opus packetization.

Each stream owns its own resampler instance; libswresample keeps filter
state between calls, so instances must never be shared across sessions.
"""

from __future__ import annotations

from fractions import Fraction

import av
from av.audio.resampler import AudioResampler


class PcmResampler:
    """Resample raw mono s16le PCM between fixed rates, streaming-safe."""

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._resampler = AudioResampler(format="s16", layout="mono", rate=dst_rate)
        self._pts = 0

    def push(self, pcm: bytes) -> bytes:
        if not pcm:
            return b""
        if self.src_rate == self.dst_rate:
            return pcm
        samples = len(pcm) // 2
        frame = av.AudioFrame(format="s16", layout="mono", samples=samples)
        frame.sample_rate = self.src_rate
        frame.time_base = Fraction(1, self.src_rate)
        frame.pts = self._pts
        self._pts += samples
        frame.planes[0].update(pcm)
        out = b""
        for resampled in self._resampler.resample(frame):
            out += bytes(resampled.planes[0])[: resampled.samples * 2]
        return out


class FrameToPcm:
    """Convert incoming aiortc audio frames to mono s16le at a target rate."""

    def __init__(self, dst_rate: int) -> None:
        self.dst_rate = dst_rate
        self._resampler = AudioResampler(format="s16", layout="mono", rate=dst_rate)

    def push(self, frame: av.AudioFrame) -> bytes:
        out = b""
        for resampled in self._resampler.resample(frame):
            out += bytes(resampled.planes[0])[: resampled.samples * 2]
        return out


class FrameChunker:
    """Re-slice an arbitrary PCM byte stream into fixed-duration frames."""

    def __init__(self, sample_rate: int, frame_ms: int) -> None:
        self.frame_bytes = sample_rate * 2 * frame_ms // 1000
        self._buffer = bytearray()

    def push(self, pcm: bytes) -> list[bytes]:
        self._buffer.extend(pcm)
        frames = []
        while len(self._buffer) >= self.frame_bytes:
            frames.append(bytes(self._buffer[: self.frame_bytes]))
            del self._buffer[: self.frame_bytes]
        return frames

    def clear(self) -> None:
        self._buffer.clear()
