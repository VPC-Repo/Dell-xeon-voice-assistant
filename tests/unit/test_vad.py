from gateway.orchestration.vad import UtteranceDetector

FRAME_MS = 20
RATE = 16000
FRAME_BYTES = RATE * 2 * FRAME_MS // 1000


class ScriptedVAD:
    """Frame classifier driven by a script of booleans."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def __call__(self, frame):
        val = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return val


def make_detector(script, **kw):
    return UtteranceDetector(is_speech=ScriptedVAD(script), sample_rate=RATE, **kw)


def run_frames(det, n, speaking=False, amplitude=8000):
    """Feed n frames of constant-amplitude samples (default: loud speech)."""
    events = []
    sample = int(amplitude).to_bytes(2, "little", signed=True)
    frame = sample * (FRAME_BYTES // 2)
    for _ in range(n):
        events.extend(det.process(frame, assistant_speaking=speaking))
    return events


def test_noise_that_fools_vad_does_not_open_an_utterance():
    # webrtcvad says "speech" but the level is room-noise quiet: no turn.
    script = [True] * 100
    det = make_detector(script, start_ms=120, utterance_min_rms=250)
    events = run_frames(det, 100, amplitude=100)
    assert events == []
    # The same VAD verdict at voice level opens an utterance.
    events = run_frames(det, 20, amplitude=8000)
    assert any(e.kind == "utterance_start" for e in events)


def test_utterance_start_end_and_preroll():
    # 10 silence, 30 speech, 40 silence
    script = [False] * 10 + [True] * 30 + [False] * 40
    det = make_detector(script, start_ms=120, end_silence_ms=400, pre_roll_ms=100)
    events = run_frames(det, len(script))
    kinds = [e.kind for e in events]
    assert kinds == ["utterance_start", "utterance"]
    utt = events[-1]
    assert utt.duration_ms > 0
    # Pre-roll means the utterance includes audio from before the trigger.
    assert len(utt.pcm) >= 30 * FRAME_BYTES


def test_short_blip_does_not_trigger():
    script = [False] * 5 + [True] * 3 + [False] * 50  # 60 ms of speech < 120 ms start
    det = make_detector(script, start_ms=120)
    events = run_frames(det, len(script))
    assert events == []


def test_barge_in_fires_once_per_playback():
    script = [True] * 100
    det = make_detector(script, barge_in_confirm_ms=100)
    det.playback_started()
    events = run_frames(det, 100, speaking=True)
    assert [e.kind for e in events if e.kind == "barge_in"] == ["barge_in"]
    # Re-arming allows the next playback to be interrupted again.
    det.playback_started()
    events = run_frames(det, 20, speaking=True)
    assert any(e.kind == "barge_in" for e in events)


def test_quiet_speech_does_not_barge_in():
    # VAD says "speech" but the level is below the echo gate: this is what
    # leaked playback looks like on a speakerphone with imperfect AEC.
    script = [True] * 100
    det = make_detector(script, barge_in_confirm_ms=100, barge_in_min_rms=350)
    det.playback_started()
    events = run_frames(det, 100, speaking=True, amplitude=150)
    assert not any(e.kind == "barge_in" for e in events)
    # The same speech at direct-voice level does interrupt.
    events = run_frames(det, 20, speaking=True, amplitude=8000)
    assert any(e.kind == "barge_in" for e in events)


def test_speculative_maybe_end_then_confirm():
    # speech, then silence long enough to confirm the end
    script = [True] * 20 + [False] * 40
    det = make_detector(script, start_ms=100, end_silence_ms=500, spec_silence_ms=250)
    events = run_frames(det, len(script))
    kinds = [e.kind for e in events]
    assert kinds == ["utterance_start", "utterance_maybe_end", "utterance"]
    maybe = events[1]
    assert maybe.pcm  # carries audio for speculative STT


def test_speculative_rollback_on_resumed_speech():
    # speech, a 300ms pause, then speech again, then the real end
    script = [True] * 20 + [False] * 15 + [True] * 15 + [False] * 40
    det = make_detector(script, start_ms=100, end_silence_ms=500, spec_silence_ms=250)
    events = run_frames(det, len(script))
    kinds = [e.kind for e in events]
    assert kinds == [
        "utterance_start", "utterance_maybe_end", "utterance_resumed",
        "utterance_maybe_end", "utterance",
    ]


def test_progress_events_during_long_utterance():
    script = [True] * 200
    det = make_detector(script, start_ms=100, progress_interval_ms=1000)
    events = run_frames(det, 200)  # 4 s of speech, no end
    progress = [e for e in events if e.kind == "utterance_progress"]
    assert len(progress) >= 3
    # Snapshots grow monotonically.
    sizes = [len(e.pcm) for e in progress]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_speculation_disabled_by_default():
    script = [True] * 20 + [False] * 40
    det = make_detector(script, start_ms=100, end_silence_ms=500)
    events = run_frames(det, len(script))
    assert [e.kind for e in events] == ["utterance_start", "utterance"]


def test_overflow_caps_utterance_length():
    script = [True] * 300
    det = make_detector(script, start_ms=40, max_utterance_ms=2000)
    events = run_frames(det, 300)
    assert any(e.kind == "utterance_overflow" for e in events)
    overflow = next(e for e in events if e.kind == "utterance_overflow")
    assert overflow.duration_ms <= 2000 + 200  # pre-roll slack
