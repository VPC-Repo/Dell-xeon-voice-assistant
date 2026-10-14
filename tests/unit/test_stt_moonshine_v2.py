"""Moonshine v2 streaming adapter: contract and threading behavior.

session.py treats an empty finalize as nothing recognized,
so what the adapter owes it is exact: feed() never blocks or raises,
partials accumulate in line_id order, finalize() forces a last pass and
returns final text, and a decode failure surfaces as "" rather than an
exception. The moonshine C library is stubbed -- these tests pin OUR
threading and state handling, not the model.
"""

from __future__ import annotations

import threading
import time
import types
from dataclasses import dataclass, field


@dataclass
class FakeLine:
    line_id: int
    text: str


@dataclass
class FakeTranscript:
    lines: list = field(default_factory=list)


class FakeStream:
    def __init__(self, owner):
        self._owner = owner
        self._listeners = []
        self.fed_samples = 0
        self.stopped = False

    def add_listener(self, cb):
        self._listeners.append(cb)

    def start(self):
        pass

    def add_audio(self, samples, sample_rate=16000):
        if self._owner.fail_on_add:
            raise RuntimeError("decode blew up")
        self.fed_samples += len(samples)
        # Emit a partial event the way real passes do, from inside add_audio.
        for cb in self._listeners:
            cb(types.SimpleNamespace(
                line=FakeLine(0, f"partial after {self.fed_samples}")))

    def update_transcription(self, flags=0):
        return FakeTranscript([FakeLine(0, self._owner.final_text)])

    def stop(self):
        self.stopped = True


class FakeTranscriber:
    MOONSHINE_FLAG_FORCE_UPDATE = 4

    def __init__(self, final_text="What can you do for me today?", fail_on_add=False):
        self.final_text = final_text
        self.fail_on_add = fail_on_add
        self.streams = []

    def create_stream(self):
        s = FakeStream(self)
        self.streams.append(s)
        return s


def make_pool(*transcribers):
    from gateway.adapters.stt_moonshine_v2 import _TranscriberPool

    it = iter(transcribers)
    pool = _TranscriberPool(lambda: next(it), len(transcribers))
    # Wait out the prewarm thread so tests are deterministic.
    deadline = time.monotonic() + 2
    while pool._built < len(transcribers) and time.monotonic() < deadline:
        time.sleep(0.005)
    return pool


def make_stream(final_text="Hello there.", fail_on_add=False):
    from gateway.adapters.stt_moonshine_v2 import MoonshineV2UtteranceStream

    tr = FakeTranscriber(final_text, fail_on_add)
    return tr, MoonshineV2UtteranceStream(make_pool(tr), 16000, 4)


PCM_20MS = (b"\x00\x01" * 320)  # 320 int16 samples


async def test_feed_then_finalize_returns_final_text():
    tr, stream = make_stream("The final transcript.")
    for _ in range(10):
        stream.feed(PCM_20MS)
    got = await stream.finalize()
    assert got == "The final transcript."
    assert tr.streams[0].fed_samples == 3200
    assert tr.streams[0].stopped


async def test_partials_update_while_feeding():
    _, stream = make_stream()
    stream.feed(PCM_20MS)
    deadline = time.monotonic() + 2
    while not stream.partial_text() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "partial after" in stream.partial_text()
    await stream.finalize()


async def test_decode_failure_surfaces_empty_not_raise():
    # session.py treats "" as nothing recognized, so the worst STT crash must
    # cost one batch decode, never the turn.
    _, stream = make_stream(fail_on_add=True)
    stream.feed(PCM_20MS)
    got = await stream.finalize()
    assert got == ""


async def test_lines_join_in_line_id_order():
    from gateway.adapters.stt_moonshine_v2 import MoonshineV2UtteranceStream

    class TwoLineStream(FakeStream):
        def add_audio(self, samples, sample_rate=16000):
            for cb in self._listeners:
                cb(types.SimpleNamespace(line=FakeLine(1, "second line.")))
                cb(types.SimpleNamespace(line=FakeLine(0, "First line,")))

    class TwoLineTranscriber(FakeTranscriber):
        def create_stream(self):
            s = TwoLineStream(self)
            self.streams.append(s)
            return s

    tr = TwoLineTranscriber()
    stream = MoonshineV2UtteranceStream(make_pool(tr), 16000, 4)
    stream.feed(PCM_20MS)
    deadline = time.monotonic() + 2
    while "First" not in stream.partial_text() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert stream.partial_text() == "First line, second line."
    await stream.finalize()


async def test_feed_after_close_is_a_noop():
    _, stream = make_stream()
    stream.close()
    await stream.finalize()
    stream.feed(PCM_20MS)  # must not raise


async def test_pool_of_one_serializes_two_streams():
    # An utterance leases an instance for its whole life; a second
    # utterance against a pool of one must WAIT, not interleave -- the C
    # library serializes on the transcriber handle, so interleaving would
    # reintroduce the measured collapse (878-1592ms tails at x2).
    from gateway.adapters.stt_moonshine_v2 import MoonshineV2UtteranceStream

    tr = FakeTranscriber("shared instance answer.")
    pool = make_pool(tr)
    a = MoonshineV2UtteranceStream(pool, 16000, 4)
    b = MoonshineV2UtteranceStream(pool, 16000, 4, lease_timeout_s=5)
    a.feed(PCM_20MS)
    b.feed(PCM_20MS)
    assert await a.finalize(timeout_s=5) == "shared instance answer."
    assert await b.finalize(timeout_s=5) == "shared instance answer."
    assert len(tr.streams) == 2  # both ran, one after the other


async def test_pool_of_two_runs_streams_on_separate_instances():
    from gateway.adapters.stt_moonshine_v2 import MoonshineV2UtteranceStream

    tr1 = FakeTranscriber("first.")
    tr2 = FakeTranscriber("second.")
    pool = make_pool(tr1, tr2)
    a = MoonshineV2UtteranceStream(pool, 16000, 4)
    b = MoonshineV2UtteranceStream(pool, 16000, 4)
    a.feed(PCM_20MS)
    b.feed(PCM_20MS)
    got = {await a.finalize(), await b.finalize()}
    assert got == {"first.", "second."}
    assert len(tr1.streams) == 1
    assert len(tr2.streams) == 1


async def test_exhausted_pool_times_out_to_empty():
    # No instance free within the lease timeout -> "" -> session.py's
    # the turn simply does not start. Never an exception, never a hang.
    from gateway.adapters.stt_moonshine_v2 import MoonshineV2UtteranceStream, _TranscriberPool

    pool = _TranscriberPool(lambda: (_ for _ in ()).throw(RuntimeError("no load")), 0)
    stream = MoonshineV2UtteranceStream(pool, 16000, 4, lease_timeout_s=0.05)
    stream.feed(PCM_20MS)
    assert await stream.finalize(timeout_s=2) == ""


async def test_instance_is_released_after_failure():
    # A decode crash must return the instance to the pool, or capacity
    # silently shrinks by one speaker per crash for the process lifetime.
    tr, stream = make_stream(fail_on_add=True)
    stream.feed(PCM_20MS)
    assert await stream.finalize() == ""
    from gateway.adapters.stt_moonshine_v2 import MoonshineV2UtteranceStream
    again = MoonshineV2UtteranceStream(make_pool(tr), 16000, 4)
    tr.fail_on_add = False
    again.feed(PCM_20MS)
    assert await again.finalize() == tr.final_text


async def test_full_queue_timeout_returns_empty_and_releases_instance():
    from gateway.adapters.stt_moonshine_v2 import MoonshineV2UtteranceStream

    decoding = threading.Event()
    resume = threading.Event()

    class SlowStream(FakeStream):
        def add_audio(self, samples, sample_rate=16000):
            super().add_audio(samples, sample_rate)
            decoding.set()
            resume.wait(timeout=2)

    class SlowTranscriber(FakeTranscriber):
        def create_stream(self):
            stream = SlowStream(self)
            self.streams.append(stream)
            return stream

    transcriber = SlowTranscriber()
    pool = make_pool(transcriber)
    stream = MoonshineV2UtteranceStream(pool, 16000, 4)
    try:
        stream.feed(PCM_20MS)
        assert decoding.wait(timeout=1)
        for _ in range(512):
            stream.feed(PCM_20MS)
        assert await stream.finalize(timeout_s=0.01) == ""
    finally:
        stream.close()
        resume.set()
    assert stream._done.wait(timeout=1)
    assert transcriber.streams[0].stopped
    leased = pool.lease(timeout_s=0.1)
    assert leased is transcriber
    pool.release(leased)
