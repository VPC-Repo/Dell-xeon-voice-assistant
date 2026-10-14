"""Voice-activity-driven utterance detection.

State machine over fixed-size PCM frames (mono s16le @ 16 kHz, 20 ms):

  IDLE --(speech for >= start_ms)--> ACTIVE
  ACTIVE --(silence for >= end_silence_ms)--> IDLE, utterance emitted

A pre-roll ring buffer keeps the audio just before the trigger so the first
syllable isn't clipped. While assistant audio is playing the same detector
doubles as the barge-in monitor: sustained speech for `barge_in_confirm_ms`
raises the barge-in flag once per playback.

The frame classifier is injectable so tests don't need webrtcvad, and so a
different VAD can be swapped in without touching the state machine.
"""

from __future__ import annotations

import array
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field


def frame_rms(frame: bytes) -> float:
    """RMS amplitude of a mono s16le frame."""
    samples = array.array("h", frame)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


@dataclass
class VADEvent:
    # "utterance_start" | "utterance" | "barge_in" | "utterance_overflow"
    # | "utterance_maybe_end" (speculative: silence reached spec_silence_ms
    #   but not yet end_silence_ms) | "utterance_resumed" (speech returned
    #   after a maybe_end -- cancel the speculation)
    kind: str
    pcm: bytes = b""
    duration_ms: int = 0


@dataclass
class UtteranceDetector:
    is_speech: Callable[[bytes], bool]
    frame_ms: int = 20
    sample_rate: int = 16000
    pre_roll_ms: int = 300
    start_ms: int = 120
    end_silence_ms: int = 550
    max_utterance_ms: int = 30000
    barge_in_confirm_ms: int = 250
    # While the assistant is speaking, interruption also requires this much
    # signal energy: leaked playback (imperfect AEC on speakerphones) is
    # usually quieter than direct speech at the mic. 0 disables the gate.
    barge_in_min_rms: int = 350
    # Starting an utterance requires this much energy too, so phone-mic
    # noise that fools webrtcvad doesn't open turns of silence (the
    # recognizer then returns an annotation and the assistant chats with
    # the room). 0 disables the gate.
    utterance_min_rms: int = 250
    # Speculative end-of-speech: fire utterance_maybe_end at this much
    # silence (before end_silence_ms confirms) so STT/LLM can start early;
    # a resumed voice fires utterance_resumed to cancel. 0 disables.
    spec_silence_ms: int = 0
    # Incremental STT: while the utterance is still running, emit
    # utterance_progress with audio-so-far every interval so transcription
    # can overlap speech. 0 disables.
    progress_interval_ms: int = 0

    _pre_roll: deque = field(init=False)
    _active: bool = field(default=False, init=False)
    _speech_run_ms: int = field(default=0, init=False)
    _silence_run_ms: int = field(default=0, init=False)
    _utterance: bytearray = field(default_factory=bytearray, init=False)
    _utterance_ms: int = field(default=0, init=False)
    _barge_run_ms: int = field(default=0, init=False)
    _barge_fired: bool = field(default=False, init=False)
    _maybe_end_fired: bool = field(default=False, init=False)
    _last_progress_ms: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._pre_roll = deque(maxlen=max(1, self.pre_roll_ms // self.frame_ms))

    @property
    def frame_bytes(self) -> int:
        return self.sample_rate * 2 * self.frame_ms // 1000

    @property
    def utterance_active(self) -> bool:
        return self._active

    def playback_started(self) -> None:
        """Re-arm barge-in for a new assistant playback."""
        self._barge_run_ms = 0
        self._barge_fired = False

    def process(self, frame: bytes, assistant_speaking: bool = False) -> list[VADEvent]:
        """Feed one frame; returns any events it produced."""
        events: list[VADEvent] = []
        speech = self.is_speech(frame)

        # Barge-in monitoring runs regardless of utterance state so the user
        # can cut in while the assistant is mid-sentence.
        if assistant_speaking and not self._barge_fired:
            loud = speech and (
                self.barge_in_min_rms <= 0 or frame_rms(frame) >= self.barge_in_min_rms
            )
            self._barge_run_ms = self._barge_run_ms + self.frame_ms if loud else 0
            if self._barge_run_ms >= self.barge_in_confirm_ms:
                self._barge_fired = True
                events.append(VADEvent(kind="barge_in"))

        if not self._active:
            self._pre_roll.append(frame)
            audible = speech and (
                self.utterance_min_rms <= 0 or frame_rms(frame) >= self.utterance_min_rms
            )
            self._speech_run_ms = self._speech_run_ms + self.frame_ms if audible else 0
            if self._speech_run_ms >= self.start_ms:
                self._active = True
                self._silence_run_ms = 0
                self._utterance = bytearray(b"".join(self._pre_roll))
                self._utterance_ms = len(self._pre_roll) * self.frame_ms
                # Carries the pre-roll so streaming ASR can seed its stream.
                events.append(
                    VADEvent(kind="utterance_start", pcm=bytes(self._utterance))
                )
            return events

        self._utterance.extend(frame)
        self._utterance_ms += self.frame_ms
        if (
            self.progress_interval_ms > 0
            and self._utterance_ms - self._last_progress_ms >= self.progress_interval_ms
        ):
            self._last_progress_ms = self._utterance_ms
            events.append(
                VADEvent(
                    kind="utterance_progress",
                    pcm=bytes(self._utterance),
                    duration_ms=self._utterance_ms,
                )
            )
        if speech:
            if self._maybe_end_fired:
                # The pause wasn't the end after all: cancel the speculation.
                self._maybe_end_fired = False
                events.append(VADEvent(kind="utterance_resumed"))
            self._silence_run_ms = 0
        else:
            self._silence_run_ms += self.frame_ms
            if (
                0 < self.spec_silence_ms <= self._silence_run_ms < self.end_silence_ms
                and not self._maybe_end_fired
            ):
                self._maybe_end_fired = True
                events.append(
                    VADEvent(
                        kind="utterance_maybe_end",
                        pcm=self._trimmed_utterance(),
                        duration_ms=self._utterance_ms,
                    )
                )

        if self._utterance_ms >= self.max_utterance_ms:
            events.append(
                VADEvent(
                    kind="utterance_overflow",
                    pcm=bytes(self._utterance),
                    duration_ms=self._utterance_ms,
                )
            )
            self._reset()
        elif self._silence_run_ms >= self.end_silence_ms:
            events.append(
                VADEvent(
                    kind="utterance",
                    pcm=self._trimmed_utterance(),
                    duration_ms=self._utterance_ms,
                )
            )
            self._reset()
        return events

    def _trimmed_utterance(self) -> bytes:
        """Utterance audio with trailing silence trimmed to a short tail."""
        keep_ms = self._utterance_ms - self._silence_run_ms + 200
        keep_bytes = max(self.frame_bytes, keep_ms * self.frame_bytes // self.frame_ms)
        return bytes(self._utterance[:keep_bytes])

    def _reset(self) -> None:
        self._last_progress_ms = 0
        self._maybe_end_fired = False
        self._active = False
        self._speech_run_ms = 0
        self._silence_run_ms = 0
        self._utterance = bytearray()
        self._utterance_ms = 0
        self._pre_roll.clear()


def make_webrtc_vad(aggressiveness: int, sample_rate: int) -> Callable[[bytes], bool]:
    """Real classifier backed by webrtcvad; import deferred for testability."""
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)

    def is_speech(frame: bytes) -> bool:
        try:
            return vad.is_speech(frame, sample_rate)
        except Exception:  # malformed frame length at stream edges
            return False

    return is_speech
