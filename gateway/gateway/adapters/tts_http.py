"""Streaming HTTP adapter for the local synthesis server.

Contract: POST {base}/synthesize with JSON {"text", "voice", "sample_rate",
"format": "pcm_s16le"} returns a chunked body of raw mono s16le PCM at the
requested rate. The bundled server uses Kokoro; another backend can use
the same contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from .base import AdapterHealth, TTSAdapter, TTSChunk

# Emit audio to the media layer in ~100 ms slices so playback can begin
# before the segment finishes synthesizing.
_CHUNK_MS = 100


class HttpStreamingTTS(TTSAdapter):
    def __init__(
        self,
        base_url: str,
        voice: str = "default",
        sample_rate: int = 24000,
        timeout_s: float = 60.0,
    ) -> None:
        self._voice = voice
        self._sample_rate = sample_rate
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s)

    async def synthesize(self, text: str, voice: str = "") -> AsyncIterator[TTSChunk]:
        body = {
            "text": text,
            "voice": voice or self._voice,
            "sample_rate": self._sample_rate,
            "format": "pcm_s16le",
        }
        chunk_bytes = int(self._sample_rate * 2 * _CHUNK_MS / 1000)
        buffer = b""
        async with self._client.stream("POST", "/synthesize", json=body) as resp:
            resp.raise_for_status()
            async for raw in resp.aiter_bytes():
                buffer += raw
                while len(buffer) >= chunk_bytes:
                    yield TTSChunk(pcm=buffer[:chunk_bytes], sample_rate=self._sample_rate)
                    buffer = buffer[chunk_bytes:]
        if len(buffer) >= 2:
            # s16le samples are 2 bytes; drop a trailing odd byte if the
            # upstream stream was cut mid-sample.
            buffer = buffer[: len(buffer) - (len(buffer) % 2)]
            yield TTSChunk(pcm=buffer, sample_rate=self._sample_rate)
        yield TTSChunk(pcm=b"", sample_rate=self._sample_rate, done=True)

    async def list_voices(self) -> dict:
        try:
            resp = await self._client.get("/voices", timeout=5.0)
            resp.raise_for_status()
            body = resp.json()
            return {
                "voices": body.get("voices", []),
                "default": body.get("default", "default"),
            }
        except Exception:  # noqa: BLE001 - voice list is best-effort metadata
            return {"voices": [], "default": self._voice}

    async def health(self) -> AdapterHealth:
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return AdapterHealth(
                name="tts", available=resp.is_success, detail=f"http {resp.status_code}"
            )
        except Exception as exc:  # noqa: BLE001
            return AdapterHealth(name="tts", available=False, detail=str(exc))

    async def close(self) -> None:
        await self._client.aclose()
