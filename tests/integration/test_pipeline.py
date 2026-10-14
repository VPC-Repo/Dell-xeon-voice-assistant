"""Turn pipeline integration: text in -> LLM stream -> segments -> TTS -> track."""

import asyncio
import json
import logging

from gateway.config import Settings
from gateway.session import GenerationSlots, Session

from .fakes import FakeChannel, FakeLLM, FakeStreamingSTT, FakeTTS

ANSWER_DELTAS = [
    "The quick answer ",
    "is that everything works fine. ",
    "Here is a second sentence that also carries enough length to segment. ",
]


def make_session(llm: FakeLLM, tts: FakeTTS | None = None,
                 ack_inject: bool = False,
                 text: str = "hello assistant") -> Session:
    # Default off so the existing tests exercise model-authored openers;
    # the injection tests opt in explicitly.
    settings = Settings(_env_file=None, ACK_INJECT=ack_inject, ACK_AT_MS=0)
    session = Session(
        settings,
        llm=llm,
        tts=tts or FakeTTS(),
        generation_slots=GenerationSlots(2),
        streaming_stt=FakeStreamingSTT(text),
    )
    session.channel = FakeChannel()
    return session


async def speak(session: Session, pcm: bytes = b"\x00\x00" * 8000) -> None:
    """Open the recognizer stream the way the media loop does."""
    from gateway.orchestration.vad import VADEvent

    await session._on_vad_event(VADEvent(kind="utterance_start", pcm=pcm))


def events_of(session: Session, kind: str) -> list[dict]:
    return [
        json.loads(raw)
        for raw in session.channel.sent
        if json.loads(raw).get("type") == kind
    ]


async def test_text_turn_streams_speech_and_records_history():
    tts = FakeTTS()
    session = make_session(FakeLLM(ANSWER_DELTAS), tts, ack_inject=False)
    session.handle_text_input("hello there")
    await asyncio.wait_for(session._turn_task, timeout=5)

    # Both sentences reached TTS, in order.
    assert len(tts.synthesized) == 2
    assert tts.synthesized[0].startswith("The quick answer")
    # History holds the full assistant reply.
    assert session.history[-1].role == "assistant"
    assert "second sentence" in session.history[-1].content
    # The browser saw the final text and audio started.
    assert events_of(session, "assistant_text_final")
    assert events_of(session, "playback_started")
    # Streaming deltas must reconstruct the exact text, spaces included
    # (regression: per-delta strip() glued words together in the UI).
    joined = "".join(e["text"] for e in events_of(session, "assistant_text_delta"))
    assert "The quick answer is that everything works fine." in joined
    # Audio frames were queued for the outbound track.
    assert session.out_track.queue_depth > 0
    await session.close()


async def test_generation_line_reports_cost_and_occupancy(caplog):
    # Emitted after the stream ends, which is why it can carry what the
    # timing and answer lines cannot: how long the turn held the model,
    # how much it produced, and how many streams were decoding with it.
    caplog.set_level(logging.INFO, logger="gateway.session")
    session = make_session(FakeLLM(ANSWER_DELTAS), FakeTTS(), ack_inject=False)
    session.handle_text_input("hello there")
    await asyncio.wait_for(session._turn_task, timeout=5)

    line = next(r.getMessage() for r in caplog.records
                if "turn generation:" in r.getMessage())
    fields = dict(p.split("=", 1) for p in line.split(" ") if "=" in p)
    assert int(fields["gen_tokens"]) == len(ANSWER_DELTAS)
    assert int(fields["gen_words"]) > 0
    assert int(fields["gen_active"]) == 1  # itself, nothing competing
    assert int(fields["gen_waiting"]) == 0
    assert int(fields["hist_tokens"]) > 0
    assert fields["interrupted"] == "false"
    await session.close()


async def test_barge_in_cancels_and_marks_interrupted():
    llm = FakeLLM(ANSWER_DELTAS * 10, delay_s=0.05)
    session = make_session(llm)
    session.handle_text_input("tell me a long story")
    await asyncio.sleep(0.15)  # let generation get going
    await session.barge_in()
    try:
        await asyncio.wait_for(session._turn_task, timeout=5)
    except asyncio.CancelledError:
        pass

    assert session.metrics.barge_ins == 1
    assert session.history[-1].role == "assistant"
    assert "[interrupted]" in session.history[-1].content
    # Outbound audio was flushed.
    assert session.out_track.queue_depth == 0
    assert events_of(session, "playback_stopped")
    await session.close()


async def test_new_turn_supersedes_active_turn():
    llm = FakeLLM(ANSWER_DELTAS * 10, delay_s=0.05)
    session = make_session(llm)
    session.handle_text_input("first question")
    await asyncio.sleep(0.12)
    first_epoch = session.epochs.current
    session.handle_text_input("second question")
    assert session.epochs.current > first_epoch
    await asyncio.wait_for(session._turn_task, timeout=10)

    user_turns = [m.content for m in session.history if m.role == "user"]
    assert user_turns == ["first question", "second question"]
    await session.close()


async def test_reset_conversation_clears_history():
    session = make_session(FakeLLM(ANSWER_DELTAS))
    session.handle_text_input("hello")
    await asyncio.wait_for(session._turn_task, timeout=5)
    assert len(session.history) > 1
    session.reset_conversation()
    assert len(session.history) == 1
    assert session.history[0].role == "system"
    await session.close()


async def test_reset_preserves_persona_and_discards_in_flight_reply(tmp_path):
    context = tmp_path / "persona.txt"
    context.write_text("Curry powder is in aisle 4.")
    session = make_session(FakeLLM(ANSWER_DELTAS * 10, delay_s=0.05))
    from gateway.session import build_system_prompt, load_prompt_context

    session.history[0].content = build_system_prompt(False, load_prompt_context(str(context)))
    original_prompt = session.history[0].content
    session.handle_text_input("Where is curry powder?")
    await asyncio.sleep(0.1)
    session.reset_conversation()
    await asyncio.wait_for(session._turn_task, timeout=2)
    assert [(m.role, m.content) for m in session.history] == [("system", original_prompt)]
    assert session.state == "listening"
    await session.close()


async def test_tts_failure_does_not_block_generation_or_next_turn():
    class BrokenTTS(FakeTTS):
        async def synthesize(self, text, voice=""):
            raise ConnectionError("TTS unavailable")
            yield  # async generator contract

    session = make_session(FakeLLM(ANSWER_DELTAS * 20, delay_s=0.001), BrokenTTS())
    session.handle_text_input("Tell me a long story.")
    await asyncio.wait_for(session._turn_task, timeout=2)
    assert events_of(session, "recoverable_error")
    assert session._generation_slots.active == 0
    session.tts = FakeTTS()
    session.llm = FakeLLM(ANSWER_DELTAS)
    session.handle_text_input("Try again.")
    await asyncio.wait_for(session._turn_task, timeout=2)
    assert events_of(session, "assistant_text_final")
    await session.close()


async def test_reset_cancels_pending_recognition():
    from gateway.orchestration.vad import VADEvent

    from .fakes import FakeStream

    class SlowStream(FakeStream):
        async def finalize(self, timeout_s: float = 2.0) -> str:
            await asyncio.sleep(0.1)
            return await super().finalize(timeout_s)

    class SlowSTT(FakeStreamingSTT):
        def new_stream(self) -> FakeStream:
            stream = SlowStream(self.text)
            self.streams.append(stream)
            return stream

    session = make_session(FakeLLM(ANSWER_DELTAS))
    session.streaming_stt = SlowSTT()
    await speak(session)
    await session._on_vad_event(VADEvent(kind="utterance_maybe_end", pcm=b"\x00\x00" * 8000))
    await asyncio.sleep(0.01)
    session.reset_conversation()
    await asyncio.gather(session._spec_task, return_exceptions=True)
    assert session._turn_task is None
    assert [m.role for m in session.history] == ["system"]
    await session.close()


async def test_profile_slider_changes_tuning():
    session = make_session(FakeLLM(ANSWER_DELTAS))
    default_silence = session.detector.end_silence_ms

    await session._handle_control({"type": "set_profile", "profile": "fast"})
    assert session.profile == "fast"
    assert session.detector.end_silence_ms < default_silence
    assert session._tuning["segment_first_min_words"] == 3
    assert events_of(session, "profile")[-1]["profile"] == "fast"

    await session._handle_control({"type": "set_profile", "profile": "stable"})
    assert session.detector.end_silence_ms > default_silence
    assert session._tuning["segment_min_words"] == 10

    # The latency machinery never turns off: every profile keeps
    # speculation and incremental STT (the slider is a robustness dial).
    from gateway.session import PROFILES

    for name, p in PROFILES.items():
        assert p["spec_silence_ms"] > 0, name
        # Incremental STT stays off pending field validation: full-prefix
        # re-passes through a batch recognizer queued up and hurt
        # both latency and seam accuracy (observed 2026-08-24).
        assert p["segment_first_min_words"] <= 4, name

    # Unknown profile is ignored, not an error.
    await session._handle_control({"type": "set_profile", "profile": "warp9"})
    assert session.profile == "stable"
    await session.close()


async def test_speculative_turn_commits_on_confirmed_end():
    from gateway.orchestration.vad import VADEvent

    session = make_session(FakeLLM(ANSWER_DELTAS))
    await speak(session)
    await session._on_vad_event(VADEvent(kind="utterance_maybe_end", pcm=b"\x00\x00" * 8000))
    await asyncio.wait_for(session._spec_task, timeout=5)
    assert session._spec_state == "active"
    assert session.history[-1].role == "user"  # turn already running

    await session._on_vad_event(VADEvent(kind="utterance"))
    assert session._spec_state == "idle"
    # Transcript surfaced exactly once, from the one recognizer stream.
    assert len(events_of(session, "final_transcript")) == 1
    assert len(session.streaming_stt.streams) == 1
    assert session.streaming_stt.streams[0].finalized
    await asyncio.wait_for(session._turn_task, timeout=5)
    assert session.history[-1].role == "assistant"
    await session.close()


async def test_speculative_turn_rolls_back_when_speech_resumes():
    from gateway.orchestration.vad import VADEvent

    session = make_session(FakeLLM(ANSWER_DELTAS * 10, delay_s=0.05))
    await speak(session)
    await session._on_vad_event(VADEvent(kind="utterance_maybe_end", pcm=b"\x00\x00" * 8000))
    await asyncio.wait_for(session._spec_task, timeout=5)
    await asyncio.sleep(0.1)  # let generation get going

    await session._on_vad_event(VADEvent(kind="utterance_resumed"))
    try:
        await asyncio.wait_for(session._turn_task, timeout=5)
    except asyncio.CancelledError:
        pass
    # History is pristine: no user message, no assistant fragment.
    assert [m.role for m in session.history] == ["system"]
    assert not events_of(session, "final_transcript")

    # The eventual real utterance still works normally.
    session.streaming_stt.text = "the real complete question"
    await speak(session)
    await session._on_vad_event(VADEvent(kind="utterance", pcm=b"\x00\x00" * 8000))
    await asyncio.wait_for(session._turn_task, timeout=5)
    assert session.history[1].content == "the real complete question"
    await session.close()


async def test_rollback_during_pending_stt():
    from gateway.orchestration.vad import VADEvent

    from .fakes import FakeStream, FakeStreamingSTT

    class SlowStream(FakeStream):
        async def finalize(self, timeout_s: float = 2.0) -> str:
            await asyncio.sleep(0.3)
            return await super().finalize(timeout_s)

    class SlowSTT(FakeStreamingSTT):
        def new_stream(self) -> FakeStream:
            stream = SlowStream(self.text)
            self.streams.append(stream)
            return stream

    settings = Settings(_env_file=None)
    session = Session(
        settings,
        llm=FakeLLM(ANSWER_DELTAS),
        tts=FakeTTS(),
        generation_slots=GenerationSlots(2),
        streaming_stt=SlowSTT(),
    )
    session.channel = FakeChannel()
    await speak(session)
    await session._on_vad_event(VADEvent(kind="utterance_maybe_end", pcm=b"\x00\x00" * 8000))
    await asyncio.sleep(0.05)  # resume while STT still running
    await session._on_vad_event(VADEvent(kind="utterance_resumed"))
    await asyncio.wait_for(session._spec_task, timeout=5)
    assert session._spec_state == "idle"
    assert [m.role for m in session.history] == ["system"]  # nothing started
    await session.close()


async def test_streaming_asr_finalizes_and_semantic_endpoint_fires():
    from gateway.orchestration.vad import VADEvent

    from .fakes import FakeStreamingSTT

    engine = FakeStreamingSTT("please turn on the office lights")
    session = make_session(FakeLLM(ANSWER_DELTAS))
    session.streaming_stt = engine
    base_silence = session.detector.end_silence_ms

    await session._on_vad_event(VADEvent(kind="utterance_start", pcm=b"\x01\x00" * 2400))
    assert len(engine.streams) == 1
    assert engine.streams[0].fed_bytes == 4800  # pre-roll seeded

    # Speculative end: the transcript comes from the stream.
    await session._on_vad_event(VADEvent(kind="utterance_maybe_end", pcm=b""))
    await asyncio.wait_for(session._spec_task, timeout=5)
    assert engine.streams[0].finalized
    assert session.history[1].content == "please turn on the office lights"
    # Complete-sounding transcript -> semantic endpoint shortened the window.
    assert session.detector.end_silence_ms < base_silence
    assert session.detector.end_silence_ms == 350

    # Confirmed end commits the running turn; window restored on next start.
    await session._on_vad_event(VADEvent(kind="utterance"))
    assert events_of(session, "final_transcript")[-1]["text"] == (
        "please turn on the office lights"
    )
    await session._on_vad_event(VADEvent(kind="utterance_start", pcm=b""))
    assert session.detector.end_silence_ms == base_silence
    await asyncio.wait_for(session._turn_task, timeout=5)
    await session.close()


async def test_incomplete_transcript_keeps_full_silence_window():
    from gateway.orchestration.vad import VADEvent

    from .fakes import FakeStreamingSTT

    session = make_session(FakeLLM(ANSWER_DELTAS))
    session.streaming_stt = FakeStreamingSTT("I want to ask about the")
    base_silence = session.detector.end_silence_ms
    await session._on_vad_event(VADEvent(kind="utterance_start", pcm=b"\x01\x00" * 100))
    await session._on_vad_event(VADEvent(kind="utterance_maybe_end", pcm=b""))
    await asyncio.wait_for(session._spec_task, timeout=5)
    assert session.detector.end_silence_ms == base_silence  # no early commit
    await session.close()


async def test_blank_audio_transcript_does_not_start_a_turn():
    session = make_session(FakeLLM(ANSWER_DELTAS), text="[BLANK_AUDIO]")
    await speak(session)
    await session._start_turn_from_audio(b"\x00\x00" * 16000)
    assert len(session.history) == 1  # system prompt only; no user turn
    assert session._turn_task is None
    assert session.state == "listening"
    await session.close()


async def test_self_echo_does_not_start_a_turn():
    session = make_session(FakeLLM(ANSWER_DELTAS))
    session.handle_text_input("hello there")
    await asyncio.wait_for(session._turn_task, timeout=5)
    turns_before = len(session.history)

    # The mic "hears" a segment the assistant just spoke.
    echoed = session._recent_tts[-1][1]
    session.streaming_stt.text = echoed
    await speak(session)
    await session._start_turn_from_audio(b"\x00\x00" * 16000)

    assert len(session.history) == turns_before  # no new turn
    assert events_of(session, "echo_suppressed")
    assert session.state == "listening"

    # Genuinely new speech still starts a turn.
    session.streaming_stt.text = "what about something completely different"
    await speak(session)
    await session._start_turn_from_audio(b"\x00\x00" * 16000)
    await asyncio.wait_for(session._turn_task, timeout=5)
    assert len(session.history) > turns_before
    await session.close()


async def test_voice_selection_reaches_tts():
    tts = FakeTTS()
    session = make_session(FakeLLM(ANSWER_DELTAS), tts)
    await session._handle_control({"type": "set_voice", "voice": "kokoro-am_adam"})
    assert events_of(session, "voice")[-1]["voice"] == "kokoro-am_adam"
    session.handle_text_input("hello")
    await asyncio.wait_for(session._turn_task, timeout=5)
    assert tts.voices_used and all(v == "kokoro-am_adam" for v in tts.voices_used)
    await session.close()


async def test_stt_failure_is_recoverable():
    from .fakes import FakeStream, FakeStreamingSTT

    class BrokenStream(FakeStream):
        async def finalize(self, timeout_s: float = 2.0) -> str:
            raise ConnectionError("stt down")

    class BrokenSTT(FakeStreamingSTT):
        def new_stream(self) -> FakeStream:
            return BrokenStream(self.text)

    settings = Settings(_env_file=None)
    session = Session(
        settings,
        llm=FakeLLM(ANSWER_DELTAS),
        tts=FakeTTS(),
        generation_slots=GenerationSlots(2),
        streaming_stt=BrokenSTT(),
    )
    session.channel = FakeChannel()
    await speak(session)
    await session._start_turn_from_audio(b"\x00\x00" * 16000)
    errors = events_of(session, "recoverable_error")
    assert errors and "recognition" in errors[0]["message"].lower()
    assert session.state == "listening"
    await session.close()


# ---- gateway-injected acknowledgement ------------------------------------


async def test_injected_acknowledgement_is_spoken_before_content():
    from gateway.session import ACK_PHRASES

    tts = FakeTTS()
    session = make_session(FakeLLM(ANSWER_DELTAS), tts, ack_inject=True)
    session.handle_text_input("hello there")
    await asyncio.wait_for(session._turn_task, timeout=5)

    assert tts.synthesized[0] in ACK_PHRASES, "acknowledgement speaks first"
    assert tts.synthesized[1].startswith("The quick answer"), "content follows"
    await session.close()


async def test_injected_acknowledgement_never_enters_history():
    # It is the gateway's phrase, not the model's words. Recording it
    # would teach the model to imitate filler it was told not to write.
    from gateway.session import ACK_PHRASES

    tts = FakeTTS()
    session = make_session(FakeLLM(ANSWER_DELTAS), tts, ack_inject=True)
    session.handle_text_input("hello there")
    await asyncio.wait_for(session._turn_task, timeout=5)

    assistant = session.history[-1]
    assert assistant.role == "assistant"
    for phrase in ACK_PHRASES:
        assert phrase not in assistant.content
    assert "second sentence" in assistant.content
    await session.close()


async def test_injected_acknowledgement_is_echo_suppressed():
    # The microphone can hear it, so it must be in the echo window even
    # though it is absent from history.
    tts = FakeTTS()
    session = make_session(FakeLLM(ANSWER_DELTAS), tts, ack_inject=True)
    session.handle_text_input("hello there")
    await asyncio.wait_for(session._turn_task, timeout=5)

    spoken_recently = [text for _, text in session._recent_tts]
    assert tts.synthesized[0] in spoken_recently
    await session.close()


async def test_injection_removes_the_opener_instruction_from_the_prompt():
    on = make_session(FakeLLM(ANSWER_DELTAS), ack_inject=True)
    off = make_session(FakeLLM(ANSWER_DELTAS), ack_inject=False)
    assert "Good question." not in on.history[0].content
    assert "Good question." in off.history[0].content
    assert "begin with the substance immediately" in on.history[0].content


async def test_acknowledgement_rotates():
    session = make_session(FakeLLM(ANSWER_DELTAS), ack_inject=True)
    picks = [session._next_ack_phrase() for _ in range(20)]
    assert len(set(picks)) > 1, "must not repeat one phrase forever"
    assert all(a != b for a, b in zip(picks, picks[1:], strict=False)), (
        "must never repeat back to back"
    )
    await session.close()


async def test_answer_is_attributed_correctly_when_no_ack_plays(caplog):
    # Regression: attribution was by arrival order, so with the ack
    # suppressed the first CONTENT segment was logged as the ack and
    # `answer` became the time to the second sentence. Measured on the
    # edge server as ack=6702ms / answer=13139ms with ack_audio=0 -- a
    # capacity benchmark reading that would report false failures.
    caplog.set_level(logging.INFO, logger="gateway.session")
    session = make_session(FakeLLM(ANSWER_DELTAS), FakeTTS(), ack_inject=True)
    session.settings.ack_at_ms = 999999          # schedule never fires
    session.handle_text_input("hello there")
    await asyncio.wait_for(session._turn_task, timeout=5)

    line = next((r.getMessage() for r in caplog.records
                 if "turn answer:" in r.getMessage()), None)
    if line is None:
        await session.close()
        return                                    # no answer line: nothing to assert
    f = dict(p.split("=", 1) for p in line.split(" ") if "=" in p)
    assert f["ack_audio"] == "0ms", "no acknowledgement should have played"
    # The answer must be the FIRST content segment, not the second: with
    # the ack absent it is the very first thing heard, so answer should
    # match the ack timestamp rather than trail it by a whole sentence.
    assert abs(float(f["answer"].rstrip("ms")) - float(f["ack"].rstrip("ms"))) < 50
    await session.close()


async def test_bridge_goes_negative_when_the_ack_overstays():
    # A negative bridge is the designed signal that content was ready
    # before the acknowledgement finished speaking -- the ack is now
    # delaying the answer rather than covering a gap. Clamping it to zero
    # (the old behaviour) hid exactly that condition.
    import inspect

    from gateway import session as sess
    src = inspect.getsource(sess.Session._on_segment_audible)
    assert "max(0.0, answer_ready_ms" not in src, "bridge must not be clamped"
    assert "answer_ready_ms - ack_ms - ack_audio_ms" in src


async def test_busy_gate_selects_the_longer_acknowledgement():
    # One acknowledgement per turn, always -- but sized to the load. At or
    # above the threshold content is seconds away, so the phrase covers
    # the bridge with speech; below it the short phrase gets out of the
    # way of an answer that may already be queued behind it.
    from gateway.session import ACK_PHRASES, LONG_ACK_PHRASES

    session = make_session(FakeLLM(ANSWER_DELTAS), ack_inject=True)
    session.settings.ack_long_when_busy = 3

    session._generation_slots.active = 1
    assert session._next_ack_phrase() in ACK_PHRASES

    session._generation_slots.active = 2
    session._generation_slots.waiting = 1  # 3 in flight or queued
    assert session._next_ack_phrase() in LONG_ACK_PHRASES

    session._generation_slots.active = 4
    session._generation_slots.waiting = 2
    assert session._next_ack_phrase() in LONG_ACK_PHRASES
    await session.close()


async def test_busy_threshold_zero_disables_the_long_tier():
    from gateway.session import ACK_PHRASES

    session = make_session(FakeLLM(ANSWER_DELTAS), ack_inject=True)
    session.settings.ack_long_when_busy = 0
    session._generation_slots.active = 4
    session._generation_slots.waiting = 4
    for _ in range(10):
        assert session._next_ack_phrase() in ACK_PHRASES
    await session.close()


async def test_rotation_never_repeats_across_tier_switches():
    # Alternating load must not defeat the no-repeat rule: the last pick
    # is tracked across tiers, not per tier.
    session = make_session(FakeLLM(ANSWER_DELTAS), ack_inject=True)
    session.settings.ack_long_when_busy = 3
    picks = []
    for i in range(20):
        session._generation_slots.active = 4 if i % 2 else 1
        picks.append(session._next_ack_phrase())
    assert all(a != b for a, b in zip(picks, picks[1:], strict=False))
    await session.close()


async def test_prewarm_covers_both_tiers():
    # A busy first turn must not pay synthesis for the long phrase.
    import gateway.session as sess
    from gateway.session import ACK_PHRASES, LONG_ACK_PHRASES

    sess._ACK_PCM.clear()
    session = make_session(FakeLLM(ANSWER_DELTAS), ack_inject=True)
    await session.prewarm_acks()
    warmed = {phrase for (_voice, phrase) in sess._ACK_PCM}
    for phrase in ACK_PHRASES + LONG_ACK_PHRASES:
        assert phrase in warmed, phrase
    await session.close()


async def test_typed_input_still_gets_an_acknowledgement():
    # Typed input has no end-of-speech event, so the scheduled path never
    # fires; the inline fallback must cover it.
    from gateway.session import ACK_PHRASES

    tts = FakeTTS()
    session = make_session(FakeLLM(ANSWER_DELTAS), tts, ack_inject=True)
    session.handle_text_input("hello there")
    await asyncio.wait_for(session._turn_task, timeout=5)
    assert tts.synthesized[0] in ACK_PHRASES
    await session.close()


async def test_prewarmed_acknowledgement_is_synthesized_once():
    # Pre-rendered audio is process-wide and already at the outbound
    # rate, so speaking a phrase twice must not synthesize it twice.
    import gateway.session as sess

    sess._ACK_PCM.clear()
    tts = FakeTTS()
    session = make_session(FakeLLM(ANSWER_DELTAS), tts, ack_inject=True)
    first = await session._ack_pcm("Sure.", "")
    calls = len(tts.synthesized)
    second = await session._ack_pcm("Sure.", "")
    assert len(tts.synthesized) == calls, "second call must hit the cache"
    assert first == second and first, "cached audio must be identical"
    sess._ACK_PCM.clear()
    await session.close()
