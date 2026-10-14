"""TTS backends.

A backend synthesizes one text segment to mono s16le PCM at the native
sample rate of the chosen voice. The app layer resamples to the requested
rate and streams.

Backends:
- kokoro: Kokoro-82M via kokoro-onnx. Non-autoregressive, so delivery is
  deterministic and short fragments synthesize cleanly -- which is what
  keeps the first word fast and makes the opener cacheable.
- mock: word-paced tones, for development and tests with nothing installed.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import math
import os
from collections import OrderedDict
from collections.abc import AsyncIterator

log = logging.getLogger(__name__)


class Backend(abc.ABC):
    name: str = "base"
    default_voice: str = "default"

    def voices(self) -> list[str]:
        return [self.default_voice]

    def resolve_voice(self, requested: str) -> str:
        """Map a requested voice to one this backend can serve."""
        if requested and requested != "default" and requested in self.voices():
            return requested
        return self.default_voice

    @abc.abstractmethod
    def sample_rate_for(self, voice: str) -> int:
        """Native sample rate of the given (resolved) voice."""

    @abc.abstractmethod
    def stream(self, text: str, voice: str) -> AsyncIterator[bytes]:
        """Yield mono s16le PCM chunks at sample_rate_for(voice)."""

    async def warm_up(self) -> None:
        """Load/synthesize the default voice so the first request is fast."""
        async for _ in self.stream("Warm up.", self.default_voice):
            pass


class MockBackend(Backend):
    name = "mock"
    _RATE = 24000

    def sample_rate_for(self, voice: str) -> int:
        return self._RATE

    async def stream(self, text: str, voice: str) -> AsyncIterator[bytes]:
        for word in text.split():
            freq = 220.0 * (1.0 + 0.4 * ((hash(word) % 7) / 7))
            n = self._RATE * max(120, min(320, 60 * len(word))) // 1000
            out = bytearray()
            for i in range(n):
                fade = min(1.0, i / 200, (n - i) / 200)
                s = int(32767 * 0.25 * fade * math.sin(2 * math.pi * freq * i / self._RATE))
                out += s.to_bytes(2, "little", signed=True)
            yield bytes(out)
            yield bytes(self._RATE * 2 * 60 // 1000)  # inter-word gap
            await asyncio.sleep(0)


class KokoroBackend(Backend):
    """Kokoro-82M via kokoro-onnx: natural speech well ahead of real time.

    StyleTTS2-derived and non-autoregressive, so delivery is deterministic
    (no per-call register lottery) and short fragments synthesize cleanly
    -- it keeps the fast first-word path. 24 kHz output.
    """

    name = "kokoro"
    _RATE = 24000
    CURATED = [
        "af_heart", "af_bella", "af_nicole", "am_adam",
        "am_michael", "bf_emma", "bm_george",
    ]

    # Segment 0 of every reply is a short opener the system prompt asks
    # for ("Sure." / "Good question."), so the same handful of phrases is
    # synthesized over and over -- and its cost lands directly on
    # end-of-speech -> first audio (measured 240-645 ms). Kokoro is
    # non-autoregressive and deterministic, so caching short phrases
    # returns byte-identical audio for a cache hit at ~0 ms.
    _CACHE_MAX_CHARS = 48
    _CACHE_MAX_ENTRIES = 64

    def __init__(self, model_path: str, voices_path: str, default: str = "af_heart") -> None:
        from kokoro_onnx import Kokoro  # lazy: only needed when selected

        self._kokoro = Kokoro(model_path, voices_path)
        self._cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        try:
            available = set(self._kokoro.get_voices())
        except Exception:  # older package versions
            available = set(self.CURATED)
        self._voices = [v for v in self.CURATED if v in available] or sorted(available)[:8]
        chosen = default if default in self._voices else self._voices[0]
        self.default_voice = f"kokoro-{chosen}"

    def voices(self) -> list[str]:
        return [f"kokoro-{v}" for v in self._voices]

    def sample_rate_for(self, voice: str) -> int:
        return self._RATE

    def _cache_key(self, text: str, speaker: str) -> tuple[str, str] | None:
        key_text = " ".join(text.split()).lower()
        if not key_text or len(key_text) > self._CACHE_MAX_CHARS:
            return None
        return (speaker, key_text)

    async def stream(self, text: str, voice: str) -> AsyncIterator[bytes]:
        import functools

        import numpy as np

        speaker = self.resolve_voice(voice).removeprefix("kokoro-")
        key = self._cache_key(text, speaker)
        chunk = self._RATE * 2 // 5  # ~200 ms slices keep the stream responsive

        if key is not None and (cached := self._cache.get(key)) is not None:
            self._cache.move_to_end(key)  # LRU
            self.cache_hits += 1
            for i in range(0, len(cached), chunk):
                yield cached[i : i + chunk]
                await asyncio.sleep(0)
            return

        loop = asyncio.get_running_loop()
        samples, sr = await loop.run_in_executor(
            None,
            functools.partial(self._kokoro.create, text, voice=speaker, lang="en-us"),
        )
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        if key is not None:
            self.cache_misses += 1
            # Only cache at the model's own rate: a variant rate would
            # make replayed chunks mismatch what the app layer expects.
            if sr == self._RATE:
                self._cache[key] = pcm
                while len(self._cache) > self._CACHE_MAX_ENTRIES:
                    self._cache.popitem(last=False)
        # sr should be 24000; if a model variant differs, the app layer
        # resamples from what we report, so rescale our claim via chunks.
        chunk = sr * 2 // 5
        for i in range(0, len(pcm), chunk):
            yield pcm[i : i + chunk]
            await asyncio.sleep(0)


def build_backend() -> Backend:
    """Kokoro when its model files are present, tones otherwise."""
    kind = os.environ.get("TTS_BACKEND", "auto").lower()
    if kind == "mock":
        return MockBackend()
    if kind not in ("auto", "kokoro"):
        raise RuntimeError(f"unknown TTS_BACKEND {kind!r} (expected kokoro or mock)")

    kokoro_dir = os.environ.get("KOKORO_MODEL_DIR", "")
    model = os.path.join(kokoro_dir, "kokoro-v1.0.onnx") if kokoro_dir else ""
    voices = os.path.join(kokoro_dir, "voices-v1.0.bin") if kokoro_dir else ""
    if model and os.path.isfile(model) and os.path.isfile(voices):
        try:
            backend = KokoroBackend(model, voices)
            default = os.environ.get("TTS_DEFAULT_VOICE", "")
            if default in backend.voices():
                backend.default_voice = default
            return backend
        except Exception as exc:  # package missing, bad model file
            if kind == "kokoro":
                raise
            log.warning("kokoro unavailable (%s); synthesizing tones", exc)
    elif kind == "kokoro":
        raise RuntimeError(
            "TTS_BACKEND=kokoro requires KOKORO_MODEL_DIR with "
            "kokoro-v1.0.onnx and voices-v1.0.bin"
        )
    else:
        log.warning("no Kokoro model found; synthesizing tones")
    return MockBackend()
