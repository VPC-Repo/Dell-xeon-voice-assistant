"""Kokoro's short-phrase cache.

Segment 0 of every reply is a short opener the system prompt asks for
("Sure." / "Good question."), and its synthesis cost lands directly on
end-of-speech -> first audio. Kokoro is non-autoregressive and
deterministic, so replaying cached PCM is byte-identical to
re-synthesizing it.
"""

from __future__ import annotations

import numpy as np
import pytest
from tts_server.backends import KokoroBackend


class FakeKokoro:
    """Stands in for kokoro_onnx.Kokoro; counts real synthesis calls."""

    def __init__(self) -> None:
        self.calls = 0

    def get_voices(self):
        return ["af_heart", "am_adam"]

    def create(self, text, voice="af_heart", lang="en-us"):
        self.calls += 1
        # Deterministic per (text, voice), like the real model.
        n = 2400 * max(1, len(text) // 4)
        seed = (hash((text, voice)) % 1000) / 10000.0
        return np.full(n, seed, dtype=np.float32), 24000


@pytest.fixture
def backend(monkeypatch):
    fake = FakeKokoro()
    monkeypatch.setattr(
        KokoroBackend, "__init__",
        lambda self, *a, **k: (
            setattr(self, "_kokoro", fake),
            setattr(self, "_voices", ["af_heart", "am_adam"]),
            setattr(self, "default_voice", "kokoro-af_heart"),
            setattr(self, "_cache", __import__("collections").OrderedDict()),
            setattr(self, "cache_hits", 0),
            setattr(self, "cache_misses", 0),
        ) and None,
    )
    b = KokoroBackend("m", "v")
    return b, fake


async def collect(backend, text, voice="kokoro-af_heart") -> bytes:
    return b"".join([c async for c in backend.stream(text, voice)])


async def test_short_phrase_is_cached_and_byte_identical(backend):
    b, fake = backend
    first = await collect(b, "Sure.")
    assert fake.calls == 1
    second = await collect(b, "Sure.")
    assert fake.calls == 1, "second call should hit the cache"
    assert second == first, "cached audio must be byte-identical"
    assert b.cache_hits == 1


async def test_cache_key_ignores_case_and_whitespace(backend):
    b, fake = backend
    await collect(b, "Sure.")
    await collect(b, "  sure.  ")
    assert fake.calls == 1


async def test_different_voices_do_not_share_cache(backend):
    b, fake = backend
    a = await collect(b, "Sure.", "kokoro-af_heart")
    c = await collect(b, "Sure.", "kokoro-am_adam")
    assert fake.calls == 2
    assert a != c, "each voice must get its own audio"


async def test_long_text_is_not_cached(backend):
    b, fake = backend
    long_text = "This is a full sentence of assistant speech well past the cap."
    await collect(b, long_text)
    await collect(b, long_text)
    assert fake.calls == 2, "long segments must not fill the cache"


async def test_cache_is_bounded(backend):
    b, fake = backend
    for i in range(KokoroBackend._CACHE_MAX_ENTRIES + 10):
        await collect(b, f"Phrase {i}.")
    assert len(b._cache) <= KokoroBackend._CACHE_MAX_ENTRIES
