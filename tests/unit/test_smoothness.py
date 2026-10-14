"""Playback smoothness: prebuffering track and segment silence trimming."""

import numpy as np
from gateway.media.tracks import (
    _SILENCE,
    BYTES_PER_FRAME,
    FIRST_PREBUFFER_MS,
    FRAME_MS,
    REBUFFER_MS,
    OutboundAudioTrack,
)
from gateway.orchestration.epochs import EpochController
from tts_server.trim import SilenceTrimmer

RATE = 24000


def silence(ms: int) -> bytes:
    return bytes(RATE * 2 * ms // 1000)


def tone(ms: int, amplitude: int = 5000) -> bytes:
    n = RATE * ms // 1000
    return (np.ones(n, dtype=np.int16) * amplitude).tobytes()


# ---- SilenceTrimmer -------------------------------------------------------


def test_trims_lead_and_tail_keeps_interior_pause():
    trimmer = SilenceTrimmer(RATE)
    src = silence(200) + tone(100) + silence(80) + tone(100) + silence(300)
    out = b""
    for i in range(0, len(src), 1000):
        out += trimmer.push(src[i : i + 1000])
    out += trimmer.flush()
    kept_ms = len(out) / 2 / RATE * 1000
    # ~100+80+100 of interior audio kept; the 200/300 pads dropped.
    assert 250 <= kept_ms <= 320
    samples = np.frombuffer(out, dtype=np.int16)
    assert abs(int(samples[0])) >= 300  # starts on audible audio


def test_all_silence_yields_nothing():
    trimmer = SilenceTrimmer(RATE)
    assert trimmer.push(silence(400)) == b""
    assert trimmer.flush() == b""


# ---- OutboundAudioTrack prebuffer ----------------------------------------


FIRST_FRAMES = FIRST_PREBUFFER_MS // FRAME_MS
REBUFFER_FRAMES = REBUFFER_MS // FRAME_MS


async def test_open_segment_prebuffers_then_plays():
    epochs = EpochController()
    track = OutboundAudioTrack(epochs)
    epoch = epochs.current

    # One frame of an open segment: not enough banked, keep silence.
    await track.push_pcm(epoch, b"\x01\x00" * (BYTES_PER_FRAME // 2))
    assert track._next_payload() == _SILENCE
    assert track.queue_depth == 1

    # Fill to the reply-opening threshold: playback starts.
    await track.push_pcm(
        epoch, b"\x01\x00" * (BYTES_PER_FRAME // 2) * (FIRST_FRAMES + 1)
    )
    assert track._next_payload() != _SILENCE
    assert track.playing


async def test_finished_short_segment_plays_immediately():
    epochs = EpochController()
    track = OutboundAudioTrack(epochs)
    epoch = epochs.current
    await track.push_pcm(epoch, b"\x01\x00" * (BYTES_PER_FRAME // 2))
    await track.end_of_utterance(epoch)  # segment complete
    assert track._next_payload() != _SILENCE


async def test_midsegment_starvation_counts_underrun_and_rebuffers():
    epochs = EpochController()
    track = OutboundAudioTrack(epochs)
    epoch = epochs.current
    await track.push_pcm(epoch, b"\x01\x00" * (BYTES_PER_FRAME // 2) * FIRST_FRAMES)
    while track.queue_depth:
        assert track._next_payload() != _SILENCE
    # Queue starved while the segment is still open -> underrun + rebuffer.
    assert track._next_payload() == _SILENCE
    assert track.underruns == 1
    # Post-underrun rebuffer uses the FULL window: even the reply-opening
    # amount is not enough now (a stall just proved the need).
    await track.push_pcm(epoch, b"\x01\x00" * (BYTES_PER_FRAME // 2) * FIRST_FRAMES)
    assert track._next_payload() == _SILENCE
    # Filling to the rebuffer window resumes playback.
    await track.push_pcm(
        epoch,
        b"\x01\x00" * (BYTES_PER_FRAME // 2) * (REBUFFER_FRAMES - FIRST_FRAMES + 1),
    )
    assert track._next_payload() != _SILENCE


async def test_flush_resets_to_reply_opening_target():
    epochs = EpochController()
    track = OutboundAudioTrack(epochs, first_prebuffer_ms=100, rebuffer_ms=240)
    epoch = epochs.current
    await track.push_pcm(epoch, b"\x01\x00" * (BYTES_PER_FRAME // 2) * 5)
    while track.queue_depth:
        track._next_payload()
    track._next_payload()  # starve -> underrun, target now 240ms
    track.flush()  # barge-in: next reply is latency-critical again
    await track.push_pcm(epoch, b"\x01\x00" * (BYTES_PER_FRAME // 2) * 6)
    assert track._next_payload() != _SILENCE  # 100ms target, not 240ms


# ---- per-segment audibility (acknowledgement vs substantive answer) -------


async def test_segments_report_when_each_becomes_audible():
    epochs = EpochController()
    track = OutboundAudioTrack(epochs)
    epoch = epochs.current
    seen: list[int] = []
    track.on_segment_audible = seen.append

    # Segment 0: the acknowledgement opener.
    await track.push_pcm(epoch, b"\x01\x00" * (BYTES_PER_FRAME // 2) * 5, 0)
    await track.end_of_utterance(epoch)
    while track.queue_depth:
        track._next_payload()
    assert seen == [0]
    assert track.segment_audio_ms(0) == 5 * FRAME_MS

    # Segment 1: the substantive answer.
    await track.push_pcm(epoch, b"\x01\x00" * (BYTES_PER_FRAME // 2) * 3, 1)
    await track.end_of_utterance(epoch)
    while track.queue_depth:
        track._next_payload()
    assert seen == [0, 1]
    assert track.segment_audio_ms(1) == 3 * FRAME_MS


async def test_segment_callback_fires_once_per_segment():
    epochs = EpochController()
    track = OutboundAudioTrack(epochs)
    epoch = epochs.current
    seen: list[int] = []
    track.on_segment_audible = seen.append
    await track.push_pcm(epoch, b"\x01\x00" * (BYTES_PER_FRAME // 2) * 6, 0)
    await track.end_of_utterance(epoch)
    while track.queue_depth:
        track._next_payload()
    assert seen == [0], "many frames, one segment, one callback"


async def test_flush_clears_segment_bookkeeping():
    epochs = EpochController()
    track = OutboundAudioTrack(epochs)
    epoch = epochs.current
    await track.push_pcm(epoch, b"\x01\x00" * (BYTES_PER_FRAME // 2) * 4, 0)
    await track.end_of_utterance(epoch)
    while track.queue_depth:
        track._next_payload()
    assert track.segment_audio_ms(0) > 0
    track.flush()
    assert track.segment_audio_ms(0) == 0
    assert track.segment_started_at == {}


async def test_begin_turn_resets_bookkeeping_without_dropping_audio():
    # A scheduled acknowledgement is pushed before the turn's epoch
    # exists, so the turn may KEEP that epoch rather than flushing it.
    # Flushing used to be what reset these counters; without begin_turn
    # the callbacks fire once per session instead of once per turn.
    epochs = EpochController()
    track = OutboundAudioTrack(epochs)
    epoch = epochs.current
    seen: list[int] = []
    track.on_segment_audible = seen.append
    fired: list[int] = []
    track.on_first_audio = lambda: fired.append(1)

    await track.push_pcm(epoch, b"\x01\x00" * (BYTES_PER_FRAME // 2) * 3, 0)
    await track.end_of_utterance(epoch)
    while track.queue_depth:
        track._next_payload()
    assert seen == [0] and fired == [1]

    # Next turn, same epoch, audio still queued behind it.
    await track.push_pcm(epoch, b"\x01\x00" * (BYTES_PER_FRAME // 2) * 2, 0)
    track.begin_turn()
    assert track.queue_depth == 2, "queued audio must survive"
    await track.end_of_utterance(epoch)
    while track.queue_depth:
        track._next_payload()
    assert seen == [0, 0], "segment 0 reports again on the new turn"
    assert fired == [1, 1], "first-audio reports again on the new turn"
