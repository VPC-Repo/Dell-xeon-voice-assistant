"""In-process mock adapters (ADAPTER_MODE=mock).

Runs the complete browser -> WebRTC -> VAD -> turn -> playback loop with no
model services: STT returns canned text for any detected utterance, the LLM
streams a canned multi-sentence answer at a realistic token rate, and TTS
renders an audible word-paced tone pattern so playback, segmentation gaps,
and barge-in can all be exercised by ear. Used for browser smoke tests
and frontend development.
"""

from __future__ import annotations

import asyncio
import math

from .base import (
    AdapterHealth,
    ChatMessage,
    LLMAdapter,
    LLMDelta,
    StreamingSTTAdapter,
    TTSAdapter,
    TTSChunk,
)

MOCK_ANSWER = (
    "This is the mock assistant speaking, so no language model is running yet. "
    "You are hearing the full streaming pipeline: voice detection, turn "
    "handling, segmentation, and audio playback. Each sentence you hear was "
    "synthesized as its own segment before the full answer finished. "
    "Try interrupting me mid-sentence to test barge-in cancellation."
)


class MockStream:
    def __init__(self, text: str) -> None:
        self.text = text
        self.fed = 0

    def feed(self, pcm: bytes) -> None:
        self.fed += len(pcm)

    def partial_text(self) -> str:
        return self.text if self.fed else ""

    async def finalize(self, timeout_s: float = 2.0) -> str:
        return self.text

    def close(self) -> None:
        return None


class MockStreamingSTT(StreamingSTTAdapter):
    """Canned recognizer: every utterance is the same text."""

    def __init__(self, text: str = "Tell me what you can do.") -> None:
        self.text = text

    def new_stream(self) -> MockStream:
        return MockStream(self.text)

    async def health(self) -> AdapterHealth:
        return AdapterHealth(name="stt", available=True)


class MockLLM(LLMAdapter):
    """Streams a canned answer word-by-word at ~20 tokens/second."""

    def __init__(self, answer: str = MOCK_ANSWER, delay_s: float = 0.05) -> None:
        self.answer = answer
        self.delay_s = delay_s

    async def stream_chat(self, messages: list[ChatMessage]):
        await asyncio.sleep(0.4)  # simulate time to first token
        for word in self.answer.split(" "):
            await asyncio.sleep(self.delay_s)
            yield LLMDelta(text=word + " ")
        yield LLMDelta(done=True, finish_reason="stop")

    async def health(self) -> AdapterHealth:
        return AdapterHealth(name="llm", available=True, detail="mock")


class MockTTS(TTSAdapter):
    """Word-paced tone bursts: audible, correctly timed, obviously synthetic.

    Each word becomes a short tone whose pitch varies, separated by brief
    silence, at roughly natural speaking pace -- enough to test playback
    ordering, inter-segment gaps, and barge-in by ear.
    """

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate

    def _tone(self, freq: float, ms: int, volume: float = 0.25) -> bytes:
        n = self.sample_rate * ms // 1000
        out = bytearray()
        for i in range(n):
            fade = min(1.0, i / 200, (n - i) / 200)  # click-free edges
            sample = int(32767 * volume * fade * math.sin(2 * math.pi * freq * i / self.sample_rate))
            out += sample.to_bytes(2, "little", signed=True)
        return bytes(out)

    async def synthesize(self, text: str, voice: str = ""):
        base = 220.0
        for i, word in enumerate(text.split()):
            freq = base * (1.0 + 0.4 * ((hash(word) % 7) / 7))
            ms = max(120, min(320, 60 * len(word)))
            yield TTSChunk(pcm=self._tone(freq, ms), sample_rate=self.sample_rate)
            yield TTSChunk(pcm=bytes(self.sample_rate * 2 * 60 // 1000), sample_rate=self.sample_rate)
            if i % 4 == 3:
                await asyncio.sleep(0)  # stay cooperative on long segments
        yield TTSChunk(pcm=b"", sample_rate=self.sample_rate, done=True)

    async def health(self) -> AdapterHealth:
        return AdapterHealth(name="tts", available=True, detail="mock")
