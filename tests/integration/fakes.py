"""In-process fake adapters for integration tests. No network, no models."""

from __future__ import annotations

import asyncio

from gateway.adapters.base import (
    AdapterHealth,
    ChatMessage,
    LLMAdapter,
    LLMDelta,
    TTSAdapter,
    TTSChunk,
)


class FakeLLM(LLMAdapter):
    def __init__(self, deltas: list[str], delay_s: float = 0.0) -> None:
        self.deltas = deltas
        self.delay_s = delay_s
        self.requests: list[list[ChatMessage]] = []

    async def stream_chat(self, messages):
        self.requests.append(messages)
        for text in self.deltas:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield LLMDelta(text=text)
        yield LLMDelta(done=True, finish_reason="stop")

    async def health(self) -> AdapterHealth:
        return AdapterHealth(name="llm", available=True)


class FakeTTS(TTSAdapter):
    def __init__(self, sample_rate: int = 24000, ms_per_segment: int = 100) -> None:
        self.sample_rate = sample_rate
        self.ms = ms_per_segment
        self.synthesized: list[str] = []
        self.voices_used: list[str] = []

    async def synthesize(self, text: str, voice: str = ""):
        self.synthesized.append(text)
        self.voices_used.append(voice)
        pcm = b"\x01\x00" * (self.sample_rate * self.ms // 1000)
        yield TTSChunk(pcm=pcm, sample_rate=self.sample_rate)
        yield TTSChunk(pcm=b"", sample_rate=self.sample_rate, done=True)

    async def health(self) -> AdapterHealth:
        return AdapterHealth(name="tts", available=True)


class FakeStream:
    """Streaming-ASR utterance stream: partials build as frames arrive."""

    def __init__(self, final_text: str) -> None:
        self.final_text = final_text
        self.fed_bytes = 0
        self.finalized = False
        self.closed = False

    def feed(self, pcm: bytes) -> None:
        self.fed_bytes += len(pcm)

    def partial_text(self) -> str:
        return self.final_text if self.fed_bytes else ""

    async def finalize(self, timeout_s: float = 2.0) -> str:
        self.finalized = True
        return self.final_text

    def close(self) -> None:
        self.closed = True


class FakeStreamingSTT:
    def __init__(self, text: str = "streamed transcript here") -> None:
        self.text = text
        self.streams: list[FakeStream] = []

    def new_stream(self) -> FakeStream:
        stream = FakeStream(self.text)
        self.streams.append(stream)
        return stream

    async def health(self) -> AdapterHealth:
        return AdapterHealth(name="stt", available=True)


class FakeChannel:
    readyState = "open"

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, data: str) -> None:
        self.sent.append(data)
