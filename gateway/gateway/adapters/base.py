"""Adapter interfaces.

The browser and media protocol must never depend on a particular STT, LLM,
or TTS implementation. Every backend is reached through one of these
interfaces; swapping a runtime means writing a new adapter, nothing else.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMDelta:
    """One normalized streaming event from the language model."""

    text: str = ""
    done: bool = False
    finish_reason: str | None = None


@dataclass
class TTSChunk:
    """Raw PCM audio from the TTS engine (mono, s16le at `sample_rate`)."""

    pcm: bytes
    sample_rate: int
    done: bool = False


@dataclass
class AdapterHealth:
    name: str
    available: bool
    detail: str = ""
    extra: dict = field(default_factory=dict)


class StreamingSTTAdapter(abc.ABC):
    """A recognizer that decodes while the person speaks.

    `new_stream()` opens one utterance; the session feeds it PCM at the
    recognizer's sample rate as frames arrive, then awaits `finalize()`
    for the text. An empty string means nothing was recognized."""

    @abc.abstractmethod
    def new_stream(self): ...

    @abc.abstractmethod
    async def health(self) -> AdapterHealth: ...

    async def close(self) -> None:  # pragma: no cover - trivial default
        return None


class LLMAdapter(abc.ABC):
    @abc.abstractmethod
    def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[LLMDelta]:
        """Stream a completion. Must be cancellable by closing the iterator."""

    @abc.abstractmethod
    async def health(self) -> AdapterHealth: ...

    async def close(self) -> None:  # pragma: no cover - trivial default
        pass


class TTSAdapter(abc.ABC):
    @abc.abstractmethod
    def synthesize(self, text: str, voice: str = "") -> AsyncIterator[TTSChunk]:
        """Stream PCM for one speakable segment. Cancellable mid-stream.

        `voice` is a backend voice name; empty means the backend default.
        """

    async def list_voices(self) -> dict:
        """Available voices; backends without a catalog report a default."""
        return {"voices": ["default"], "default": "default"}

    @abc.abstractmethod
    async def health(self) -> AdapterHealth: ...

    async def close(self) -> None:  # pragma: no cover - trivial default
        pass
