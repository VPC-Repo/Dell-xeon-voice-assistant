"""Per-session state and the turn pipeline.

One Session owns: a WebRTC peer connection, a control data channel, one
epoch controller, conversation history, the VAD/utterance detector, and the
outbound audio track. Nothing in here is shared between sessions except the
adapter clients (stateless HTTP) and the generation semaphore.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .adapters.base import ChatMessage, LLMAdapter, TTSAdapter
from .config import Settings
from .media.resample import FrameChunker, FrameToPcm, PcmResampler
from .media.tracks import SAMPLE_RATE as OUT_RATE
from .media.tracks import OutboundAudioTrack
from .orchestration.echo import is_probable_echo
from .orchestration.endpoint import looks_complete
from .orchestration.epochs import EpochController
from .orchestration.normalize import (
    StageTagStreamFilter,
    ThinkTagStreamFilter,
    strip_reasoning,
    strip_stage_tags,
    strip_stt_annotations,
)
from .orchestration.segmenter import Segment, SpeechSegmenter
from .orchestration.vad import UtteranceDetector, VADEvent, make_webrtc_vad

log = logging.getLogger(__name__)


@dataclass
class SlotEntry:
    """Occupancy of the generation gate at the moment a turn was admitted."""

    active: int  # generations running, including this one
    waiting: int  # other turns still queued for a slot
    limit: int


class GenerationSlots:
    """Limit concurrent generations and record occupancy for latency logs."""

    def __init__(self, limit: int) -> None:
        self._sem = asyncio.Semaphore(limit)
        self.limit = limit
        self.active = 0
        self.waiting = 0
        self.peak_active = 0

    async def __aenter__(self) -> SlotEntry:
        self.waiting += 1
        queued_with = self.waiting - 1  # others already waiting, not us
        try:
            await self._sem.acquire()
        finally:
            self.waiting -= 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        return SlotEntry(active=self.active, waiting=queued_with, limit=self.limit)

    async def __aexit__(self, *exc_info) -> bool:
        self.active -= 1
        self._sem.release()
        return False


# Ask the model for an opener only when the gateway does not inject one.
# Both variants request conversational prose suited to speech synthesis.
_PROMPT_OPENER = (
    "Begin every reply with a complete opening sentence of two to four "
    "words ending in a period, such as 'Sure.' or 'Good question.', then "
    "continue in"
)
_PROMPT_NO_OPENER = (
    "Do not open with a filler acknowledgement such as 'Sure' or 'Of "
    "course' -- begin with the substance immediately, in"
)
_PROMPT_BODY = (
    " natural, flowing conversational prose: complete sentences of "
    "varied length that connect ideas the way a person talks. Never chop an "
    "answer into a series of terse fragments, headline-style lines, or a spoken "
    "list; weave items into sentences instead. Answer directly, in a single "
    "paragraph of a few sentences unless the user asks for detail. Do not use "
    "Markdown, tables, bullet symbols, line breaks, raw URLs, citation markers, "
    "or code fences in spoken replies. Write only the words to be spoken: never "
    "stage directions or bracketed cues such as [pause] or [sigh] -- pauses come "
    "from punctuation alone. Never expose hidden reasoning. When a long "
    "procedure is necessary, introduce it briefly and offer to continue step by "
    "step."
)
_PROMPT_HEAD = (
    "You are a private spoken assistant. Respond for listening, not for reading. "
)


def load_prompt_context(path: str) -> str:
    """Deployment-specific grounding appended to the system prompt.

    A demo persona lives in a text file, not in code: a store's aisle
    map, a clinic's hours, whatever facts the assistant must answer
    from instead of hallucinating. Kept small (~400 tokens) so prefill
    stays cheap and SGLang's prefix cache covers it across turns. A
    missing or unreadable file degrades to the plain assistant with a
    warning -- never a failed session.
    """
    if not path:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        log.warning("prompt context file %s unreadable (%s); continuing without", path, exc)
        return ""
    if not text:
        return ""
    return "\n\n" + text


def build_system_prompt(model_writes_opener: bool, context: str = "") -> str:
    lead = _PROMPT_OPENER if model_writes_opener else _PROMPT_NO_OPENER
    return _PROMPT_HEAD + lead + _PROMPT_BODY + context


# Spoken while the model works. Deliberately content-free: they are
# chosen before the answer exists, so they must be true of any reply.
# Short and few, so they stay resident in the TTS phrase cache and cost
# ~0 ms to speak.
ACK_PHRASES = (
    "Sure.",
    "Let me see.",
    "Good question.",
    "One moment.",
    "Okay, let me check.",
    "Right, let me think.",
    "Let me think about that.",
    "Sure, one moment.",
)

# Longer variants for when the generation gate is busy and content will
# be late. Still exactly ONE acknowledgement per turn at the same fixed
# schedule -- the phrase is sized to the expected wait, so the bridge is
# covered by informative speech instead of silence. Adapting the PHRASE
# rather than the schedule keeps the ack deterministic and keeps
# pre-ack silence (the expensive kind: the user does not yet know they
# were heard) at its minimum; sliding the schedule later would only
# trade cheap post-ack silence for expensive pre-ack silence. Bounded at
# ~2s of audio so a misprediction (content arriving early under a
# falling load) delays it by at most about a second.
LONG_ACK_PHRASES = (
    "Sure, give me a moment to look at that.",
    "Good question, let me think that through.",
    "Okay, give me a second to check on that.",
    "Right, let me take a look at that for you.",
    "Sure, let me work through that one.",
)

# Segment id for the injected acknowledgement. Negative so it can never
# collide with the segmenter's own ids, which start at 0.
ACK_SEGMENT_ID = -1

# Pre-rendered acknowledgement audio, keyed (voice, phrase) and stored
# already resampled to the outbound rate. Speaking one then costs a dict
# lookup and a memcpy: no synthesis, no HTTP hop to the TTS server, no
# resampling on the critical path. Process-wide, so every session shares
# it, and warmed in the background when a session picks a voice.
_ACK_PCM: dict[tuple[str, str], bytes] = {}
_ACK_PCM_LOCK = asyncio.Lock()

# Bounded handoff between the segmenter and TTS; backpressure, never growth.
SEGMENT_QUEUE_SIZE = 8

# Profiles for the UI's stable <-> fast slider. The latency machinery
# (speculative turn start, incremental STT, fast first segment) is safe
# under false triggers -- speculation rolls back cleanly -- so it stays ON
# at every position. The slider is a NOISE/ECHO ROBUSTNESS dial: energy
# gates, barge-in confirmation, and end-of-speech patience.
PROFILES: dict[str, dict[str, int]] = {
    "stable": {
        "spec_silence_ms": 350,
        "utterance_min_rms": 500,
        "vad_end_silence_ms": 700,
        "barge_in_confirm_ms": 600,
        "barge_in_min_rms": 900,
        "segment_min_words": 10,
        "segment_min_chars": 50,
        "segment_first_min_words": 4,
        "segment_first_min_chars": 16,
        "segment_max_wait_ms": 2000,
    },
    "balanced": {
        "spec_silence_ms": 300,
        "utterance_min_rms": 250,
        "vad_end_silence_ms": 550,
        "barge_in_confirm_ms": 250,
        "barge_in_min_rms": 350,
        "segment_min_words": 8,
        "segment_min_chars": 40,
        "segment_first_min_words": 4,
        "segment_first_min_chars": 16,
        "segment_max_wait_ms": 1500,
    },
    "fast": {
        "spec_silence_ms": 250,
        # Echo floors stay meaningful even on Fast: energy gates cost
        # headphone users nothing (direct voice is loud) but stop the
        # assistant from hearing itself on speakers.
        "utterance_min_rms": 300,
        "vad_end_silence_ms": 400,
        "barge_in_confirm_ms": 300,
        "barge_in_min_rms": 500,
        "segment_min_words": 6,
        "segment_min_chars": 30,
        "segment_first_min_words": 3,
        "segment_first_min_chars": 12,
        "segment_max_wait_ms": 900,
    },
}


@dataclass
class SessionMetrics:
    turns: int = 0
    barge_ins: int = 0
    cancellations: int = 0
    last_eos_to_first_audio_ms: float | None = None
    errors: int = 0
    underruns: int = 0  # folded from the outbound track on session close


class Session:
    def __init__(
        self,
        settings: Settings,
        llm: LLMAdapter,
        tts: TTSAdapter,
        generation_slots: GenerationSlots,
        streaming_stt=None,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.settings = settings
        self.llm = llm
        self.tts = tts
        self.streaming_stt = streaming_stt  # the streaming recognizer, one stream per utterance
        self._asr_stream = None  # per-utterance streaming decoder
        self._generation_slots = generation_slots

        self.epochs = EpochController()
        self.out_track = OutboundAudioTrack(
            self.epochs,
            first_prebuffer_ms=settings.playback_first_prebuffer_ms,
            rebuffer_ms=settings.playback_rebuffer_ms,
        )
        self.out_track.on_first_audio = self._on_first_audible
        self.out_track.on_segment_audible = self._on_segment_audible
        self.ack_inject = settings.ack_inject
        self._ack_phrases = tuple(
            p.strip() for p in settings.ack_phrases.split(",") if p.strip()
        ) or ACK_PHRASES
        self._ack_phrases_busy = tuple(
            p.strip() for p in settings.ack_phrases_busy.split(",") if p.strip()
        ) or LONG_ACK_PHRASES
        self._last_ack = ""
        self._ack_task: asyncio.Task | None = None
        self._ack_scheduled = False  # a timed ack owns this turn
        self._content_started = False  # first real segment already queued
        self._ack_epoch: int | None = None
        self._prewarm_task: asyncio.Task | None = None
        self.history: list[ChatMessage] = [
            ChatMessage(
                "system",
                build_system_prompt(
                    not self.ack_inject,
                    load_prompt_context(settings.prompt_context_file),
                ),
            )
        ]
        self._history_reset_epoch = -1
        self.metrics = SessionMetrics()
        self.created_at = time.monotonic()
        self.pc = None  # set by signaling
        self.channel = None  # control data channel, set on open
        self.muted = False
        self.state = "connecting"
        self.closed = False

        self.detector = UtteranceDetector(
            is_speech=make_webrtc_vad(settings.vad_aggressiveness, settings.stt_sample_rate),
            sample_rate=settings.stt_sample_rate,
            pre_roll_ms=settings.vad_pre_roll_ms,
            start_ms=settings.vad_start_ms,
            end_silence_ms=settings.vad_end_silence_ms,
            max_utterance_ms=int(settings.max_utterance_seconds * 1000),
            barge_in_confirm_ms=settings.barge_in_confirm_ms,
            barge_in_min_rms=settings.barge_in_min_rms,
            utterance_min_rms=settings.utterance_min_rms,
            spec_silence_ms=settings.speculative_silence_ms,
        )
        self._turn_task: asyncio.Task | None = None
        self._reader_task: asyncio.Task | None = None
        self._eos_at: float | None = None  # end-of-speech timestamp for latency metric

        # Per-turn stage timings, logged at first audio so real sessions
        # show where the milliseconds went.
        self._last_stt_ms = 0.0
        self._tts_ready_ms = 0.0
        self._ack_ms = 0.0        # eos -> first audible word (the opener)
        self._ack_eos_at: float | None = None
        self._ack_text = ""       # what the opener actually said
        self._answered = False    # substantive segment already reported
        # When the first CONTENT pcm reached the track -- i.e. when the
        # answer COULD have been heard. It becomes audible later if the
        # acknowledgement is still draining, so measuring the audible
        # moment charges the pipeline for the ack's own playback.
        self._content_ready_at: float | None = None
        self._turn_started_at = 0.0
        self._ttft_ms = 0.0
        self._gen_queue_ms = 0.0
        self._first_seg_ms = 0.0
        self._first_seg_tokens = 0  # tokens consumed before segment 0 emitted
        # Generation occupancy and cost, known only once the stream ends.
        # These say how long a turn HOLDS the model, which the latency
        # numbers do not: a slot is held for the whole generation, so a
        # long answer keeps competing for memory bandwidth well after its
        # own latency-critical part is over.
        self._gen_ms = 0.0
        self._gen_tokens = 0
        self._gen_active = 0
        self._gen_waiting = 0

        # Semantic endpointing: base end-silence to restore each utterance,
        # and bookkeeping for false-cutoff detection.
        self._end_silence_base = settings.vad_end_silence_ms
        self._semantic_committed_at = 0.0

        # Per-session voice; empty = TTS backend default.
        self.voice = ""

        # Speculative turn state machine. STT is slower than the window
        # between maybe-end and confirmed end, so speculation runs as a
        # background task and the states track the interleavings:
        #   idle -> pending (maybe_end fired, STT running)
        #        -> active (STT done, turn generating) or
        #           committed (real end arrived while STT still running)
        #   any state -> idle on rollback (speech resumed) or completion.
        self._spec_state = "idle"
        self._spec_text = ""
        self._spec_task: asyncio.Task | None = None
        self._discard_turn_epoch: int | None = None

        # Incremental STT: prefix of the in-progress utterance already
        # transcribed in the background, so end-of-speech only pays for
        # the tail (decoded with the prefix as context).

        # What the assistant spoke recently, for self-echo suppression:
        # (monotonic timestamp, segment text).
        self._recent_tts: deque[tuple[float, str]] = deque(maxlen=32)

        # Per-session tuning overlay (UI slider). Starts from .env values;
        # apply_profile() replaces the group wholesale.
        self.profile = "balanced"
        self._tuning = {
            "segment_min_words": settings.segment_min_words,
            "segment_min_chars": settings.segment_min_chars,
            "segment_first_min_words": settings.segment_first_min_words,
            "segment_first_min_chars": settings.segment_first_min_chars,
            "segment_max_wait_ms": settings.segment_max_wait_ms,
        }

    def apply_profile(self, name: str) -> None:
        profile = PROFILES.get(name)
        if profile is None:
            return
        self.profile = name
        self._end_silence_base = profile["vad_end_silence_ms"]
        self.detector.end_silence_ms = profile["vad_end_silence_ms"]
        self.detector.barge_in_confirm_ms = profile["barge_in_confirm_ms"]
        self.detector.barge_in_min_rms = profile["barge_in_min_rms"]
        self.detector.utterance_min_rms = profile["utterance_min_rms"]
        self.detector.spec_silence_ms = profile["spec_silence_ms"]
        for key in self._tuning:
            self._tuning[key] = profile[key]
        log.info("session %s: profile -> %s", self.id, name)
        self.send_event("profile", profile=name)

    # ---- events to the browser ---------------------------------------------

    def send_event(self, kind: str, **payload) -> None:
        if self.channel is None or self.channel.readyState != "open":
            return
        try:
            self.channel.send(json.dumps({"type": kind, **payload}))
        except Exception:  # channel torn down mid-send
            log.debug("session %s: dropped event %s", self.id, kind)

    def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.send_event("state", state=state)

    # ---- inbound media -----------------------------------------------------

    def start_media(self, inbound_track) -> None:
        self._reader_task = asyncio.create_task(self._read_inbound(inbound_track))

    async def _read_inbound(self, track) -> None:
        to_pcm = FrameToPcm(self.settings.stt_sample_rate)
        chunker = FrameChunker(self.settings.stt_sample_rate, self.detector.frame_ms)
        self.set_state("listening")
        try:
            while not self.closed:
                frame = await track.recv()
                if self.muted:
                    chunker.clear()
                    continue
                for vad_frame in chunker.push(to_pcm.push(frame)):
                    for event in self.detector.process(
                        vad_frame, assistant_speaking=self.out_track.playing
                    ):
                        await self._on_vad_event(event)
                    # Streaming ASR eats every in-utterance frame as it
                    # arrives (the utterance_start event seeded the
                    # pre-roll; events above may have opened the stream).
                    if self._asr_stream is not None and self.detector.utterance_active:
                        self._asr_stream.feed(vad_frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # track ended or decode failure
            if not self.closed:
                log.info("session %s: inbound media ended: %s", self.id, exc)

    async def _on_vad_event(self, event: VADEvent) -> None:
        if event.kind == "barge_in":
            await self.barge_in()
        elif event.kind == "utterance_start":
            if self._ack_task and not self._ack_task.done():
                self._ack_task.cancel()  # stale schedule from a prior turn
            self._reset_turn_audio_state()
            # Restore the full end-silence window (a previous turn's
            # semantic endpoint may have shortened it).
            self.detector.end_silence_ms = self._end_silence_base
            # False-cutoff signal: new speech right after a semantic early
            # commit means we probably cut the user off mid-turn.
            if (
                self._semantic_committed_at
                and time.monotonic() - self._semantic_committed_at < 2.0
            ):
                log.warning("session %s: possible false cutoff after semantic commit", self.id)
            if self.streaming_stt is not None:
                if self._asr_stream is not None:
                    self._asr_stream.close()
                self._asr_stream = self.streaming_stt.new_stream()
                self._asr_stream.feed(event.pcm)  # pre-roll
            self.set_state("transcribing")
        elif event.kind == "utterance_maybe_end":
            # Silence has held long enough to bet the turn is over: start
            # STT (and then the LLM) in the background, overlapping the
            # rest of the end-silence wait. Runs as a task so this media
            # loop keeps watching for resumed speech.
            # The user actually stopped speaking spec_silence ago; anchor
            # the latency clock there so eos_to_audio is honest.
            self._eos_at = time.monotonic() - self.detector.spec_silence_ms / 1000
            self._spec_state = "pending"
            if self.ack_inject and self.settings.ack_at_ms > 0:
                self._ack_scheduled = True
                self._ack_task = asyncio.create_task(self._speak_timed_ack())
            self._spec_task = asyncio.create_task(self._speculative_start(event.pcm))
        elif event.kind == "utterance_resumed":
            self._rollback_speculation()
        elif event.kind == "utterance":
            if self._spec_state == "active":
                # The bet paid off and the turn is already generating;
                # just surface the transcript to the browser.
                self._spec_state = "idle"
                self.send_event("final_transcript", text=self._spec_text)
                return
            if self._spec_state == "pending":
                # Real end arrived while speculative STT is still running;
                # let that STT finalize the turn (no second transcription).
                self._spec_state = "committed"
                return
            self._eos_at = time.monotonic() - self.detector.end_silence_ms / 1000
            await self._start_turn_from_audio(event.pcm)
            self._drop_asr_stream()
        elif event.kind == "utterance_overflow":
            self.send_event(
                "recoverable_error",
                message="Utterance exceeded the maximum length and was truncated.",
            )
            self._eos_at = time.monotonic()
            await self._start_turn_from_audio(event.pcm)

    # ---- turn entry points -------------------------------------------------

    def _drop_asr_stream(self) -> None:
        if self._asr_stream is not None:
            self._asr_stream.close()
            self._asr_stream = None

    async def _transcribe_utterance(self, pcm: bytes) -> str:
        """Finalize the utterance's recognizer stream and return its text.

        The stream has been decoding since the first frame, so this costs
        the finalization only. An empty result means nothing was
        recognized and the turn does not start."""
        if self._asr_stream is None:
            log.warning("session %s: no recognizer stream for this utterance", self.id)
            return ""
        stream, self._asr_stream = self._asr_stream, None
        return strip_stt_annotations(await stream.finalize())

    async def _start_turn_from_audio(self, pcm: bytes) -> None:
        stt_t0 = time.monotonic()
        try:
            text = await self._transcribe_utterance(pcm)
            self._last_stt_ms = (time.monotonic() - stt_t0) * 1000
        except Exception as exc:
            self.metrics.errors += 1
            log.warning("session %s: STT failed: %s", self.id, exc)
            self.send_event("recoverable_error", message="Speech recognition is unavailable.")
            self.set_state("listening")
            return
        if not text:
            self.set_state("listening")
            return
        if self._is_self_echo(text):
            log.info("session %s: suppressed self-echo: %r", self.id, text[:80])
            self.send_event("echo_suppressed", text=text)
            self.set_state("listening")
            return
        self.send_event("final_transcript", text=text)
        self.start_turn(text)

    async def _speculative_start(self, pcm: bytes) -> None:
        """Start recognition while the end-of-speech window is still pending."""
        stt_t0 = time.monotonic()
        try:
            text = await self._transcribe_utterance(pcm)
            self._last_stt_ms = (time.monotonic() - stt_t0) * 1000
        except Exception as exc:
            log.warning("session %s: speculative STT failed: %s", self.id, exc)
            text = ""
        state = self._spec_state
        if state not in ("pending", "committed"):
            return  # rolled back while STT was running
        if not text or self._is_self_echo(text):
            self._spec_state = "idle"
            if state == "committed":
                # The confirmed utterance was empty/echo: same outcome the
                # normal path would have produced.
                self.set_state("listening")
            return
        self._spec_text = text
        log.info("session %s: speculative turn start: %r", self.id, text[:60])
        if state == "committed":
            self._spec_state = "idle"
            self.send_event("final_transcript", text=text)
        else:
            self._spec_state = "active"
            # Semantic endpoint: a complete-sounding transcript doesn't
            # need the full end-silence window -- commit the turn sooner.
            semantic = self.settings.semantic_end_silence_ms
            if semantic > 0 and looks_complete(text):
                self.detector.end_silence_ms = min(self._end_silence_base, semantic)
                self._semantic_committed_at = time.monotonic()
                log.info(
                    "session %s: semantic endpoint: complete, end_silence -> %dms",
                    self.id, self.detector.end_silence_ms,
                )
        self.start_turn(text)

    async def _ack_pcm(self, phrase: str, voice: str) -> bytes:
        """Outbound-rate PCM for one phrase, synthesized at most once."""
        key = (voice, phrase)
        cached = _ACK_PCM.get(key)
        if cached is not None:
            return cached
        async with _ACK_PCM_LOCK:
            cached = _ACK_PCM.get(key)  # another task may have won the race
            if cached is not None:
                return cached
            resampler = PcmResampler(self.settings.tts_sample_rate, OUT_RATE)
            parts: list[bytes] = []
            async for chunk in self.tts.synthesize(phrase, voice):
                if chunk.pcm:
                    parts.append(resampler.push(chunk.pcm))
            pcm = b"".join(parts)
            if pcm:
                _ACK_PCM[key] = pcm
            return pcm

    async def prewarm_acks(self) -> None:
        """Render every acknowledgement for this session's voice.

        Runs in the background at session start so the first turn never
        pays synthesis for a phrase that will be spoken hundreds of times.
        """
        if not self.ack_inject:
            return
        voice = self.voice
        phrases = self._ack_phrases + self._ack_phrases_busy
        for phrase in phrases:
            if self.closed:
                return
            try:
                await self._ack_pcm(phrase, voice)
            except Exception as exc:  # noqa: BLE001 - warming is best-effort
                log.debug("session %s: ack prewarm failed (%s)", self.id, exc)
                return
        log.info(
            "session %s: %d acknowledgements ready for %s",
            self.id, len(phrases), voice or "(default)",
        )

    async def _speak_timed_ack(self) -> None:
        """Speak the acknowledgement on a fixed schedule from end of speech.

        Deliberately not gated on recognition: nothing about the phrase
        depends on the transcript, so waiting for STT only inherits its
        tail. Firing late-but-inside-budget is also what keeps the phrase
        short -- the later it starts, the less audio is needed to reach
        the first content.
        """
        if self._eos_at is None:
            return
        delay = self.settings.ack_at_ms / 1000 - (time.monotonic() - self._eos_at)
        if delay > 0:
            await asyncio.sleep(delay)
        # "listening" means recognition already came back empty or as our
        # own echo and the turn was abandoned -- acknowledging nothing.
        if self.closed or self.state == "listening":
            return
        # The answer beat the schedule. An acknowledgement now would land
        # in the middle of it, and is redundant anyway: content arriving
        # this fast IS the acknowledgement.
        if self._content_started:
            return
        phrase = self._next_ack_phrase()
        try:
            pcm = await self._ack_pcm(phrase, self.voice)
        except Exception as exc:  # noqa: BLE001 - never break a turn for this
            log.warning("session %s: acknowledgement failed: %s", self.id, exc)
            return
        if not pcm or self.closed:
            return
        epoch = self.epochs.current
        self._ack_epoch = epoch
        self._ack_text = phrase
        self.detector.playback_started()
        self.set_state("speaking")
        self.send_event("playback_started")
        await self.out_track.push_pcm(epoch, pcm, ACK_SEGMENT_ID)
        await self.out_track.end_of_utterance(epoch)
        self._recent_tts.append((time.monotonic(), phrase))

    def _reset_turn_audio_state(self) -> None:
        """Clear per-turn playback bookkeeping at the START of an utterance.

        Not in start_turn: a scheduled acknowledgement is spoken before
        generation begins, so resetting there would wipe the very state
        it had just recorded.
        """
        self._ack_ms = 0.0
        self._ack_eos_at = None
        self._ack_text = ""
        self._answered = False
        self._content_started = False
        self._content_ready_at = None
        self.out_track.begin_turn()

    def _ack_load(self) -> int:
        """Generations in flight or queued when the acknowledgement fires."""
        slots = self._generation_slots
        return slots.active + slots.waiting

    def _next_ack_phrase(self) -> str:
        """Pick an acknowledgement sized to the load, never repeating.

        Above the busy threshold content will be seconds away, so a longer
        phrase covers more of the bridge with speech; below it the short
        phrase gets out of content's way (an over-long acknowledgement
        DELAYS an answer that is already queued behind it).
        """
        threshold = self.settings.ack_long_when_busy
        busy = threshold > 0 and self._ack_load() >= threshold
        tier = self._ack_phrases_busy if busy else self._ack_phrases
        options = [p for p in tier if p != self._last_ack]
        phrase = random.choice(options or list(tier))
        self._last_ack = phrase
        return phrase

    def _rollback_speculation(self) -> None:
        """The user kept talking: undo the speculation, whatever its stage."""
        state = self._spec_state
        if state == "idle":
            return
        self._spec_state = "idle"
        log.info("session %s: speculation rolled back (%s)", self.id, state)
        # The speaker only paused: do not talk over them.
        if self._ack_task and not self._ack_task.done():
            self._ack_task.cancel()
        self._ack_scheduled = False
        if state == "active":
            # A turn is generating: mark it discard-on-finish and cancel.
            # The cancelled task's _finish_turn cleans the history.
            self._discard_turn_epoch = self.epochs.current
            self.epochs.advance()
            self.out_track.flush()
            if self._turn_task and not self._turn_task.done():
                self._turn_task.cancel()
        self.set_state("transcribing")  # the utterance is still going

    def _is_self_echo(self, transcript: str) -> bool:
        """Did the mic just hear our own playback? (Speakerphone leakage.)"""
        window = self.settings.echo_suppress_window_s
        cutoff = time.monotonic() - window
        recent = [text for ts, text in self._recent_tts if ts >= cutoff]
        return is_probable_echo(
            transcript, recent, self.settings.echo_suppress_similarity
        )

    def handle_text_input(self, text: str) -> None:
        text = text.strip()[: self.settings.max_text_input_chars]
        if not text:
            return
        self._reset_turn_audio_state()
        self._eos_at = time.monotonic()
        self.send_event("final_transcript", text=text)
        self.start_turn(text)

    def start_turn(self, user_text: str) -> None:
        """Cancel anything in flight and launch a new generation."""
        self._turn_started_at = time.monotonic()
        self._ttft_ms = 0.0
        self._gen_queue_ms = 0.0
        self._first_seg_ms = 0.0
        self._gen_ms = 0.0
        self._gen_tokens = 0
        self._gen_active = 0
        self._gen_waiting = 0
        if self._ack_epoch is not None and self.epochs.is_current(self._ack_epoch):
            # The acknowledgement for THIS turn is already playing; keep
            # its epoch so flushing does not cut it off mid-word.
            epoch = self._ack_epoch
        else:
            epoch = self.epochs.advance()
            self.out_track.flush()
        self._ack_epoch = None
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        self.history.append(ChatMessage("user", user_text))
        self._trim_history()
        self._turn_task = asyncio.create_task(self._run_turn(epoch))

    # ---- the turn pipeline -------------------------------------------------

    async def _run_turn(self, epoch: int) -> None:
        self.metrics.turns += 1
        self.set_state("thinking")
        # Whitespace-preserving filters for the display stream: reasoning
        # spans and stage cues ([pause]) are dropped even when a tag is
        # split across token deltas.
        think_filter = ThinkTagStreamFilter()
        stage_filter = StageTagStreamFilter()
        segmenter = SpeechSegmenter(
            min_words=self._tuning["segment_min_words"],
            min_chars=self._tuning["segment_min_chars"],
            max_chars=self.settings.segment_max_chars,
            first_min_words=self._tuning["segment_first_min_words"],
            first_min_chars=self._tuning["segment_first_min_chars"],
            first_fast_open=not self.ack_inject,
        )
        segment_queue: asyncio.Queue[Segment | None] = asyncio.Queue(SEGMENT_QUEUE_SIZE)
        spoken: list[Segment] = []
        raw_text_parts: list[str] = []
        inject_inline = self.ack_inject and not self._ack_scheduled
        self._ack_scheduled = False

        async def produce() -> None:
            nonlocal interrupted
            # Queue the acknowledgement before waiting for a generation slot.
            if inject_inline:
                phrase = self._next_ack_phrase()
                await segment_queue.put(
                    Segment(segment_id=ACK_SEGMENT_ID, text=phrase, raw=phrase)
                )
            try:
                # Time spent waiting for a generation slot (admission gate).
                # Part of user-facing ttft; logged separately so queueing is
                # distinguishable from LLM prefill when tuning
                # MAX_ACTIVE_GENERATIONS.
                gen_q_t0 = time.monotonic()
                async with self._generation_slots as slot:
                    self._gen_queue_ms = (time.monotonic() - gen_q_t0) * 1000
                    self._gen_active = slot.active
                    self._gen_waiting = slot.waiting
                    gen_t0 = time.monotonic()
                    if not self.epochs.is_current(epoch):
                        raise asyncio.CancelledError
                    last_emit = time.monotonic()
                    stream = self.llm.stream_chat(list(self.history))
                    async for delta in stream:
                        if not self.epochs.is_current(epoch):
                            interrupted = True
                            break
                        if delta.text:
                            if not raw_text_parts and self._turn_started_at:
                                self._ttft_ms = (time.monotonic() - self._turn_started_at) * 1000
                            raw_text_parts.append(delta.text)
                            display = stage_filter.feed(think_filter.feed(delta.text))
                            if display:
                                self.send_event("assistant_text_delta", text=display)
                            for seg in segmenter.feed(delta.text):
                                if seg.segment_id == 0 and self._turn_started_at:
                                    self._first_seg_ms = (
                                        time.monotonic() - self._turn_started_at
                                    ) * 1000
                                    self._first_seg_tokens = len(raw_text_parts)
                                await segment_queue.put(seg)
                                last_emit = time.monotonic()
                        stale_ms = (time.monotonic() - last_emit) * 1000
                        if stale_ms > self._tuning["segment_max_wait_ms"]:
                            for seg in segmenter.flush_stale():
                                await segment_queue.put(seg)
                                last_emit = time.monotonic()
                        if delta.done:
                            break
                    if not interrupted:
                        for seg in segmenter.finish():
                            await segment_queue.put(seg)
                    self._gen_ms = (time.monotonic() - gen_t0) * 1000
                    # Stream deltas, which SGLang emits one token at a time;
                    # the same approximation `seg_tokens` already reports.
                    self._gen_tokens = len(raw_text_parts)
            except Exception as exc:
                self.metrics.errors += 1
                log.warning("session %s: LLM stream failed: %s", self.id, exc)
                self.send_event("recoverable_error", message="The assistant is unavailable.")
                raise
            await segment_queue.put(None)

        interrupted = False
        try:
            # Failure or cancellation of either stage cancels the other, even
            # when a bounded queue is full. Both tasks finish before returning.
            async with asyncio.TaskGroup() as group:
                group.create_task(self._speak_segments(epoch, segment_queue, spoken))
                group.create_task(produce())
        except asyncio.CancelledError:
            interrupted = True
        except ExceptionGroup:
            interrupted = True

        self._finish_turn(epoch, raw_text_parts, spoken, interrupted)

    def _finish_turn(
        self,
        epoch: int,
        raw_text_parts: list[str],
        spoken: list[Segment],
        interrupted: bool,
    ) -> None:
        if epoch <= self._history_reset_epoch:
            return
        if self._discard_turn_epoch == epoch:
            # Rolled-back speculation: erase the turn entirely -- drop the
            # speculative user message, record nothing for the assistant.
            self._discard_turn_epoch = None
            if self.history and self.history[-1].role == "user":
                self.history.pop()
            return
        full_text = strip_stage_tags(strip_reasoning("".join(raw_text_parts)))
        if interrupted or not self.epochs.is_current(epoch):
            # Keep only what the user actually heard.
            heard = " ".join(seg.text for seg in spoken).strip()
            content = (heard + " [interrupted]").strip() if heard else "[interrupted]"
            self.history.append(ChatMessage("assistant", content))
            self.metrics.cancellations += 1
        else:
            self.history.append(ChatMessage("assistant", full_text))
            self.send_event("assistant_text_final", text=full_text)
        self._log_generation(full_text, interrupted)
        self._trim_history()
        if self.epochs.is_current(epoch):
            self.set_state("listening")

    async def _speak_segments(
        self, epoch: int, queue: asyncio.Queue, spoken: list[Segment]
    ) -> None:
        """Consume ordered segments, synthesize, and stream to the track."""
        resampler = PcmResampler(self.settings.tts_sample_rate, OUT_RATE)
        first_audio_sent = False
        while True:
            seg = await queue.get()
            if seg is None:
                break
            if not self.epochs.is_current(epoch):
                continue  # drain remaining segments without synthesizing
            if seg.segment_id == (ACK_SEGMENT_ID if self.ack_inject else 0):
                self._ack_text = seg.text
            try:
                async for chunk in self.tts.synthesize(seg.text, self.voice):
                    if not self.epochs.is_current(epoch):
                        break
                    if chunk.pcm:
                        pcm48 = resampler.push(chunk.pcm)
                        if seg.segment_id != ACK_SEGMENT_ID:
                            if not self._content_started:
                                self._content_ready_at = time.monotonic()
                            self._content_started = True
                        if not first_audio_sent and pcm48:
                            first_audio_sent = True
                            self._note_tts_ready()
                            self.detector.playback_started()
                            self.set_state("speaking")
                            self.send_event("playback_started")
                        await self.out_track.push_pcm(epoch, pcm48, seg.segment_id)
                await self.out_track.end_of_utterance(epoch)
                if self.epochs.is_current(epoch):
                    # The injected phrase is ours, not the model's: it must
                    # not enter conversation history. It DOES go in the
                    # echo window, since the microphone can hear it.
                    if seg.segment_id != ACK_SEGMENT_ID:
                        spoken.append(seg)
                    self._recent_tts.append((time.monotonic(), seg.text))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.metrics.errors += 1
                log.warning("session %s: TTS failed on segment %s: %s", self.id, seg.segment_id, exc)
                self.send_event("recoverable_error", message="Speech synthesis is unavailable.")
                raise

    def _note_tts_ready(self) -> None:
        """TTS handed over the first PCM of this turn (not yet audible)."""
        if self._eos_at is not None:
            self._tts_ready_ms = (time.monotonic() - self._eos_at) * 1000

    def _on_segment_audible(self, segment_id: int) -> None:
        """A segment just started playing. Separates two user-facing events.

        With gateway injection, the acknowledgement uses ACK_SEGMENT_ID;
        otherwise the model's opener uses segment 0. Report the opener,
        first substantive content, and the gap between them separately.
        """
        if self._answered:
            return
        now = time.monotonic()
        # Keep our own origin: the turn-timing log clears _eos_at as soon
        # as the first frame plays, but the substantive segment arrives
        # later and must be measured from the same zero.
        if self._ack_eos_at is None:
            if self._eos_at is None:
                return
            self._ack_eos_at = self._eos_at
        # Identify the acknowledgement by its SEGMENT ID, not by arrival
        # order. Order-based attribution silently mislabels every metric
        # whenever no acknowledgement plays (ACK_AT_MS past the turn, or a
        # skip because content beat the schedule): the first content
        # segment was recorded as the ack and `answer` became the time to
        # the SECOND sentence, roughly doubling it.
        ack_id = ACK_SEGMENT_ID if self.ack_inject else 0
        if segment_id == ack_id:
            self._ack_ms = (now - self._ack_eos_at) * 1000
            return
        # First substantive segment.
        self._answered = True
        answer_ms = (now - self._ack_eos_at) * 1000
        # What the pipeline actually owes: when content was ready to play.
        # If the acknowledgement was still draining, the audible moment is
        # later by exactly that overhang -- a cost of the ack's length, not
        # of the machine, and one a capacity benchmark should not be
        # charged for.
        answer_ready_ms = answer_ms
        if self._content_ready_at is not None:
            answer_ready_ms = (self._content_ready_at - self._ack_eos_at) * 1000
        ack_ms = self._ack_ms or answer_ms
        ack_audio_ms = self.out_track.segment_audio_ms(ack_id)
        # Silence the user sat through after the opener finished, measured
        # against readiness so it can go NEGATIVE: that means content was
        # ready before the acknowledgement finished speaking, and the ack
        # is now delaying the answer rather than covering a gap. A
        # persistently negative bridge is the signal to shorten or
        # interrupt it.
        bridge_ms = answer_ready_ms - ack_ms - ack_audio_ms
        # The LLM stage timings are still zero when an INJECTED
        # acknowledgement plays -- it is spoken before generation starts --
        # so they are reported here, where they are known, rather than on
        # the turn-timing line.
        tts_ms = 0.0
        if self._turn_started_at and self._first_seg_ms:
            tts_ms = (
                (time.monotonic() - self._turn_started_at) * 1000 - self._first_seg_ms
            )
        log.info(
            "session %s turn answer: ack=%.0fms ack_audio=%.0fms bridge=%.0fms "
            "answer=%.0fms answer_ready=%.0fms ttft=%.0fms first_seg=%.0fms seg_tokens=%d "
            "tts_first=%.0fms injected=%s opener=%r profile=%s voice=%s",
            self.id, ack_ms, ack_audio_ms, bridge_ms, answer_ms, answer_ready_ms,
            self._ttft_ms, self._first_seg_ms, self._first_seg_tokens, tts_ms,
            "true" if self.ack_inject else "false",
            self._ack_text[:40], self.profile, self.voice or "(default)",
        )
        self.send_event(
            "metrics",
            ack_ms=round(ack_ms),
            answer_ms=round(answer_ms),
            bridge_ms=round(bridge_ms),
        )

    def _on_first_audible(self) -> None:
        """Measure latency when the first audio frame is sent for playback."""
        if self._eos_at is None:
            return
        ms = (time.monotonic() - self._eos_at) * 1000
        self.metrics.last_eos_to_first_audio_ms = ms
        self.send_event("metrics", eos_to_first_audio_ms=round(ms))
        tts_ms = 0.0
        if self._turn_started_at and self._first_seg_ms:
            tts_ms = (
                (time.monotonic() - self._turn_started_at) * 1000 - self._first_seg_ms
            )
        log.info(
            "session %s turn timing: stt=%.0fms gen_q=%.0fms "
            "ttft=%.0fms first_seg=%.0fms seg_tokens=%d "
            "tts_first=%.0fms tts_ready=%.0fms eos_to_audio=%.0fms "
            "profile=%s voice=%s hist=%d",
            self.id, self._last_stt_ms, self._gen_queue_ms,
            self._ttft_ms, self._first_seg_ms, self._first_seg_tokens,
            tts_ms, self._tts_ready_ms, ms,
            self.profile, self.voice or "(default)", len(self.history),
        )
        self._eos_at = None

    # ---- interruption and control -----------------------------------------

    async def barge_in(self) -> None:
        self.metrics.barge_ins += 1
        await self._cancel_generation()
        self.set_state("listening")

    async def stop_response(self) -> None:
        await self._cancel_generation()
        self.set_state("listening")

    async def _cancel_generation(self) -> None:
        """Atomic cancellation: epoch first, then flush every stage."""
        self.epochs.advance()
        self._cancel_pending_recognition()
        dropped = self.out_track.flush()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        self.send_event("playback_stopped", dropped_frames=dropped)

    def _cancel_pending_recognition(self) -> None:
        self._spec_state = "idle"
        self._ack_epoch = None
        self._ack_scheduled = False
        self._drop_asr_stream()
        for task in (self._spec_task, self._ack_task):
            if task and task is not asyncio.current_task() and not task.done():
                task.cancel()

    def reset_conversation(self) -> None:
        self._history_reset_epoch = self.epochs.current
        self.epochs.advance()
        self._cancel_pending_recognition()
        self.out_track.flush()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        self.history = self.history[:1]
        self.set_state("listening")

    def _log_generation(self, full_text: str, interrupted: bool) -> None:
        """Report what a turn cost the model, once the stream has ended.

        First-audio timings are logged while generation is running. This
        records the complete generation duration and slot occupancy.

        `hist_tokens` uses the same 4-chars-per-token approximation as
        `_trim_history`, so the two agree about when trimming will fire.
        """
        hist_chars = sum(len(m.content) for m in self.history)
        rate = self._gen_tokens / (self._gen_ms / 1000) if self._gen_ms else 0.0
        log.info(
            "session %s turn generation: gen=%.0fms gen_tokens=%d "
            "gen_words=%d gen_rate=%.1f gen_active=%d gen_waiting=%d "
            "gen_limit=%d hist_tokens=%d hist_msgs=%d interrupted=%s",
            self.id, self._gen_ms, self._gen_tokens,
            len(full_text.split()), rate, self._gen_active, self._gen_waiting,
            self._generation_slots.limit, hist_chars // 4, len(self.history),
            "true" if interrupted else "false",
        )

    def _trim_history(self) -> None:
        """Bound history by a rough token budget (4 chars ~ 1 token).

        When over budget, trim to half the budget in one pass. This reduces
        how often the prompt prefix changes and invalidates the KV cache.
        """
        budget_chars = self.settings.llm_history_max_tokens * 4
        total = sum(len(m.content) for m in self.history)
        if total <= budget_chars:
            return
        target = budget_chars // 2
        while total > target and len(self.history) > 3:
            removed = self.history.pop(1)  # keep the system prompt
            total -= len(removed.content)

    # ---- control channel ---------------------------------------------------

    def attach_channel(self, channel) -> None:
        self.channel = channel

        @channel.on("message")
        def on_message(raw) -> None:
            try:
                msg = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                return
            if isinstance(msg, dict):
                asyncio.ensure_future(self._handle_control(msg))

        self.send_event("state", state=self.state)
        self._prewarm_task = asyncio.ensure_future(self.prewarm_acks())

    async def _handle_control(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "mute":
            self.muted = True
        elif kind == "unmute":
            self.muted = False
        elif kind == "stop_response":
            await self.stop_response()
        elif kind == "reset_conversation":
            self.reset_conversation()
        elif kind == "text_input":
            self.handle_text_input(str(msg.get("text", "")))
        elif kind == "set_profile":
            self.apply_profile(str(msg.get("profile", "")))
        elif kind == "client_metrics":
            # Browser-clock truth: last voiced mic sample -> first played
            # assistant sample, plus WebRTC transport quality.
            def num(key):
                v = msg.get(key)
                return v if isinstance(v, (int, float)) else -1

            log.info(
                "session %s client metrics: e2e=%.0fms rtt=%.0fms jitter=%.0fms "
                "jbuf=%.0fms lost=%.0f relay=%s",
                self.id, num("e2e_ms"), num("rtt_ms"), num("jitter_ms"),
                num("jbuf_ms"), num("lost"), bool(msg.get("relay")),
            )
        elif kind == "set_voice":
            # Applies from the next segment; validation happens at the TTS
            # server, which falls back to its default for unknown names.
            self.voice = str(msg.get("voice", ""))[:80]
            log.info("session %s: voice -> %s", self.id, self.voice or "(default)")
            self.send_event("voice", voice=self.voice)
            # Acknowledgements are per-voice; render the new set now so the
            # next turn does not pay synthesis on the critical path.
            self._prewarm_task = asyncio.ensure_future(self.prewarm_acks())

    # ---- lifecycle ---------------------------------------------------------

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._drop_asr_stream()
        self.epochs.advance()
        self.out_track.flush()
        tasks = [task for task in (
            self._turn_task, self._reader_task, self._ack_task, self._prewarm_task,
            self._spec_task,
        ) if task is not None and task is not asyncio.current_task()]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.out_track.stop()
        if self.pc is not None:
            await self.pc.close()


class SessionManager:
    def __init__(
        self,
        settings: Settings,
        llm: LLMAdapter,
        tts: TTSAdapter,
        streaming_stt=None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.tts = tts
        self.streaming_stt = streaming_stt
        self.sessions: dict[str, Session] = {}
        self.generation_slots = GenerationSlots(settings.max_active_generations)
        self.rejected_sessions = 0
        # Counters folded in from closed sessions, so /api/metrics reports
        # lifetime totals rather than only currently-live sessions.
        self.closed_totals = SessionMetrics()

    def _fold(self, session: Session) -> None:
        m = session.metrics
        self.closed_totals.turns += m.turns
        self.closed_totals.barge_ins += m.barge_ins
        self.closed_totals.cancellations += m.cancellations
        self.closed_totals.errors += m.errors
        self.closed_totals.underruns += session.out_track.underruns

    def create(self) -> Session | None:
        self._reap_expired()
        if len(self.sessions) >= self.settings.max_sessions:
            self.rejected_sessions += 1
            return None
        session = Session(
            self.settings, self.llm, self.tts, self.generation_slots,
            streaming_stt=self.streaming_stt,
        )
        self.sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    async def close_session(self, session_id: str) -> bool:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        self._fold(session)
        await session.close()
        return True

    def _reap_expired(self) -> None:
        now = time.monotonic()
        expired = [
            sid
            for sid, s in self.sessions.items()
            if now - s.created_at > self.settings.max_session_seconds or s.closed
        ]
        for sid in expired:
            session = self.sessions.pop(sid)
            self._fold(session)
            asyncio.ensure_future(session.close())

    async def close_all(self) -> None:
        for sid in list(self.sessions):
            await self.close_session(sid)
