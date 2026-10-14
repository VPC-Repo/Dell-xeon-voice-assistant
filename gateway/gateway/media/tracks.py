"""Outbound audio track: a paced, epoch-aware PCM player.

The TTS pipeline pushes 48 kHz mono s16le PCM tagged with a generation
epoch. `recv()` emits fixed 20 ms frames at wall-clock rate, playing queued
audio when available and comfort silence otherwise, so the RTP timestamp
sequence never breaks. Stale-epoch audio is dropped at the queue head --
this is where barge-in actually silences the speaker.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from fractions import Fraction

import av
from aiortc import MediaStreamTrack

SAMPLE_RATE = 48000
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2
_SILENCE = bytes(BYTES_PER_FRAME)

# Bound outbound buffering to ~10 s of audio; beyond that the TTS producer
# blocks (backpressure) instead of letting latency grow without limit.
MAX_QUEUED_FRAMES = 500

# De-jitter: while a segment is still being synthesized, hold playback
# until enough audio is queued so a momentary TTS stall doesn't punch
# 20 ms silence holes into words. A finished segment plays immediately.
# Two targets: the reply-opening fill is charged directly to
# end-of-speech -> first audio, so it uses a smaller window; rebuffering
# after a real underrun uses the full window, since protection has just
# proven necessary. Overridable per deployment (PLAYBACK_*_MS in .env).
FIRST_PREBUFFER_MS = 120
REBUFFER_MS = 240


class OutboundAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(
        self,
        epoch_controller,
        first_prebuffer_ms: int = FIRST_PREBUFFER_MS,
        rebuffer_ms: int = REBUFFER_MS,
    ) -> None:
        super().__init__()
        self._epochs = epoch_controller
        self._first_frames = max(1, first_prebuffer_ms // FRAME_MS)
        self._rebuffer_frames = max(1, rebuffer_ms // FRAME_MS)
        self._queue: deque[tuple[int, int, bytes]] = deque()
        self._space = asyncio.Event()
        self._space.set()
        self._pending = bytearray()
        self._pending_epoch = -1
        self._start: float | None = None
        self._pts = 0
        self.playing = False  # true while real (non-silence) audio is flowing
        self._segment_open = False  # producer still synthesizing this segment
        self._buffering = True  # hold playback until prebuffer fills
        self._buffer_target = self._first_frames
        self.underruns = 0  # mid-segment queue starvations (jerkiness metric)
        # Callback when the first real audio frame leaves the track.
        self.on_first_audio = None
        self._announced = False
        self._pending_segment = 0
        # Per-segment playback facts, cleared each turn by flush(): when
        # each segment first became audible, and how many frames of it
        # have played. Together these separate "silence the user waited
        # through" from "audio the user was listening to".
        self.segment_started_at: dict[int, float] = {}
        self.segment_frames: dict[int, int] = {}
        self.on_segment_audible = None

    async def push_pcm(self, epoch: int, pcm: bytes, segment_id: int = 0) -> None:
        """Queue 48 kHz mono s16le PCM for playback. Applies backpressure.

        Frames carry their segment id so playback can report when each
        segment actually becomes audible -- the acknowledgement opener and
        the substantive answer that follows it are different events with
        different latency budgets.
        """
        self._segment_open = True
        self._pending_epoch = epoch
        self._pending_segment = segment_id
        self._pending.extend(pcm)
        while len(self._pending) >= BYTES_PER_FRAME:
            await self._put(epoch, bytes(self._pending[:BYTES_PER_FRAME]), segment_id)
            del self._pending[:BYTES_PER_FRAME]

    async def end_of_utterance(self, epoch: int) -> None:
        """Flush the sub-frame remainder, zero-padded to a full frame."""
        if self._pending and self._pending_epoch == epoch:
            frame = bytes(self._pending) + _SILENCE[len(self._pending) :]
            await self._put(epoch, frame, self._pending_segment)
        self._pending.clear()
        self._segment_open = False

    async def _put(self, epoch: int, frame: bytes, segment_id: int) -> None:
        while len(self._queue) >= MAX_QUEUED_FRAMES:
            self._space.clear()
            await self._space.wait()
            if not self._epochs.is_current(epoch):
                return  # cancelled while blocked; drop silently
        self._queue.append((epoch, segment_id, frame))

    def begin_turn(self) -> None:
        """Reset per-turn playback bookkeeping WITHOUT dropping audio.

        A scheduled acknowledgement is pushed before the turn's epoch
        exists, so the turn may keep that epoch rather than flushing it.
        Reset the callbacks here so each turn reports its first audio and
        segment timings, including turns that retain an acknowledgement.
        """
        self._announced = False
        self.segment_started_at = {}
        self.segment_frames = {}

    def flush(self) -> int:
        """Drop everything queued (barge-in / stop). Returns frames dropped."""
        dropped = len(self._queue)
        self._queue.clear()
        self._pending.clear()
        self._space.set()
        self.playing = False
        self._segment_open = False
        self._buffering = True
        self._buffer_target = self._first_frames  # next fill opens a reply
        self.begin_turn()
        return dropped

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    def segment_audio_ms(self, segment_id: int) -> float:
        """Milliseconds of segment `segment_id` actually played so far."""
        return self.segment_frames.get(segment_id, 0) * FRAME_MS

    def _next_payload(self) -> bytes:
        # Drop stale-epoch frames at the head first.
        while self._queue and not self._epochs.is_current(self._queue[0][0]):
            self._queue.popleft()
            self._space.set()
        if not self._queue:
            if self.playing and self._segment_open:
                # Starved mid-segment: rebuffer instead of drip-feeding
                # audio through 20 ms silence holes. Use the full window;
                # a stall just demonstrated the need for it.
                self.underruns += 1
                self._buffering = True
                self._buffer_target = self._rebuffer_frames
            self.playing = False
            return _SILENCE
        if self._buffering:
            # A segment still synthesizing must bank the target first; a
            # finished segment (or filled buffer) plays out immediately.
            if self._segment_open and len(self._queue) < self._buffer_target:
                return _SILENCE
            self._buffering = False
        _, seg_id, frame = self._queue.popleft()
        self._space.set()
        self.playing = True
        self.segment_frames[seg_id] = self.segment_frames.get(seg_id, 0) + 1
        if seg_id not in self.segment_started_at:
            self.segment_started_at[seg_id] = time.monotonic()
            if self.on_segment_audible is not None:
                try:
                    self.on_segment_audible(seg_id)
                except Exception:  # noqa: BLE001 - metrics must not break audio
                    pass
        if not self._announced:
            self._announced = True
            if self.on_first_audio is not None:
                try:
                    self.on_first_audio()
                except Exception:  # noqa: BLE001 - metrics must not break audio
                    pass
        return frame

    async def recv(self) -> av.AudioFrame:
        # Pace to wall clock so RTP timestamps stay continuous.
        if self._start is None:
            self._start = time.monotonic()
        target = self._start + (self._pts + SAMPLES_PER_FRAME) / SAMPLE_RATE
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

        payload = self._next_payload()
        frame = av.AudioFrame(format="s16", layout="mono", samples=SAMPLES_PER_FRAME)
        frame.planes[0].update(payload)
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = Fraction(1, SAMPLE_RATE)
        self._pts += SAMPLES_PER_FRAME
        return frame
