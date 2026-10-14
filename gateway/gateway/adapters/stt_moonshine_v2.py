"""Streaming ASR via Moonshine v2 (moonshine-voice, sliding-window encoder).

Each concurrent utterance leases its own transcriber instance. Instances
warm in a background thread and use one ONNX Runtime decode thread by
default to avoid CPU contention. If a lease or finalization times out,
the stream returns no text and the gateway does not start a turn.

Contract: `new_stream()` per utterance; `feed(pcm)` never blocks the
media loop (worker thread); `finalize()` returns the final text; empty
results mean nothing was recognized and session.py starts no turn.

The runtime is moonshine-voice's own C library (bundled ONNX Runtime),
with official quantized .ort weights fetched from download.moonshine.ai
into `cache_root` on first use. English models are MIT-licensed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading

from .base import AdapterHealth, StreamingSTTAdapter

log = logging.getLogger(__name__)

_END = object()

class _TranscriberPool:
    """Pre-warmed single-threaded transcriber instances, leased per utterance.

    Sized to the concurrent-session cap: a session speaks at most one
    utterance at a time, so `size` instances can never be outnumbered.
    Loading costs ~0.9 s each, so instances are built in a background
    thread; an utterance that starts before warm-up reaches its slot
    waits in lease() up to the timeout and then falls back.
    """

    def __init__(self, factory, size: int) -> None:
        self._factory = factory
        self._queue: queue.Queue = queue.Queue()
        self._size = size
        self._built = 0
        threading.Thread(target=self._prewarm, name="moonshine-v2-prewarm",
                         daemon=True).start()

    def _prewarm(self) -> None:
        for i in range(self._size):
            try:
                self._queue.put(self._factory())
                self._built += 1
            except Exception as exc:
                log.warning("moonshine v2 instance %d failed to load: %s", i, exc)
        log.info("moonshine v2 pool warm: %d/%d instances", self._built, self._size)

    def lease(self, timeout_s: float):
        try:
            return self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def release(self, transcriber) -> None:
        self._queue.put(transcriber)


class MoonshineV2StreamingSTT(StreamingSTTAdapter):
    def __init__(
        self,
        arch: str = "medium-streaming",
        cache_root: str = "models/moonshine-v2",
        sample_rate: int = 16000,
        instances: int = 6,
        lease_timeout_s: float = 2.0,
    ) -> None:
        # Set before loading the C library to limit decoder CPU contention.
        # Operators can override it through the environment.
        os.environ.setdefault("MOONSHINE_ORT_SINGLE_THREAD", "1")
        # Lazy: only needed when this engine is enabled.
        import moonshine_voice as mv
        from moonshine_voice.transcriber import Transcriber

        self.sample_rate = sample_rate
        self._lease_timeout_s = lease_timeout_s
        self._flag_force = Transcriber.MOONSHINE_FLAG_FORCE_UPDATE
        model_path, model_arch = mv.get_model_for_language(
            "en", mv.string_to_model_arch(arch), cache_root=cache_root
        )
        self._pool = _TranscriberPool(
            lambda: Transcriber(model_path, model_arch), max(1, instances)
        )
        log.info("moonshine v2 %s: warming %d single-thread instances (%s)",
                 arch, max(1, instances), model_path)

    async def health(self) -> AdapterHealth:
        return AdapterHealth(
            name="stt", available=self._pool._built > 0,
            detail=f"moonshine v2 {self._pool._built}/{self._pool._size} instances warm",
        )

    def new_stream(self) -> "MoonshineV2UtteranceStream":
        return MoonshineV2UtteranceStream(
            self._pool, self.sample_rate, self._flag_force, self._lease_timeout_s
        )


class MoonshineV2UtteranceStream:
    def __init__(
        self,
        pool: _TranscriberPool,
        sample_rate: int,
        flag_force: int,
        lease_timeout_s: float = 2.0,
    ) -> None:
        self._pool = pool
        self._sample_rate = sample_rate
        self._flag_force = flag_force
        self._lease_timeout_s = lease_timeout_s
        self._queue: queue.Queue = queue.Queue(maxsize=512)
        self._partial = ""
        self._lines: dict[int, str] = {}
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._ending = threading.Event()
        self._closed = threading.Event()
        self._failed = False
        self._thread = threading.Thread(
            target=self._run, name="moonshine-v2-asr", daemon=True
        )
        self._thread.start()

    def _on_event(self, event) -> None:
        """Listener: track lines by id; events fire inside add_audio passes."""
        line = getattr(event, "line", None)
        if line is None or not getattr(line, "text", ""):
            return
        with self._lock:
            self._lines[line.line_id] = line.text
            self._partial = " ".join(
                t for _, t in sorted(self._lines.items()) if t
            ).strip()

    @staticmethod
    def _text_of(transcript) -> str:
        lines = getattr(transcript, "lines", None) or []
        return " ".join(
            ln.text for ln in lines if getattr(ln, "text", "")
        ).strip()

    def _run(self) -> None:
        import numpy as np

        # Leased in the worker, not in new_stream(): new_stream() runs on
        # the media event loop at utterance start, and a cold pool must
        # cost this utterance a timeout, never the loop a stall.
        transcriber = self._pool.lease(self._lease_timeout_s)
        if transcriber is None:
            log.warning("moonshine v2: no instance free; nothing recognized this turn")
            self._done.set()
            return
        stream = None
        try:
            if self._closed.is_set():
                return
            stream = transcriber.create_stream()
            stream.add_listener(self._on_event)
            stream.start()
            while not self._closed.is_set():
                if self._ending.is_set() and self._queue.empty():
                    break
                item = self._queue.get()
                if item is _END:
                    break
                samples = (
                    np.frombuffer(item, dtype=np.int16).astype(np.float32) / 32768.0
                )
                # add_audio runs the incremental decode itself, on an
                # adaptive cadence (each pass must cover at least its own
                # cost). Partials arrive through the listener; no separate
                # update call is needed while audio flows.
                stream.add_audio(samples.tolist(), sample_rate=self._sample_rate)
            if not self._closed.is_set():
                transcript = stream.update_transcription(self._flag_force)
                with self._lock:
                    self._partial = self._text_of(transcript)
        except Exception as exc:  # decode failure: surface empty, no turn
            self._failed = True
            log.warning("moonshine v2 stream failed: %s", exc)
        finally:
            if stream is not None:
                try:
                    stream.stop()
                except Exception as exc:
                    log.warning("moonshine v2 stream cleanup failed: %s", exc)
            self._pool.release(transcriber)
            self._done.set()

    def feed(self, pcm: bytes) -> None:
        """Non-blocking; drops the frame if the worker is hopelessly behind."""
        if self._done.is_set() or self._ending.is_set() or self._closed.is_set():
            return
        try:
            self._queue.put_nowait(pcm)
        except queue.Full:
            log.warning("moonshine v2 stream backlogged; dropping frame")

    def partial_text(self) -> str:
        with self._lock:
            return self._partial.strip()

    async def finalize(self, timeout_s: float = 2.0) -> str:
        """Signal end of audio and return the final transcript."""
        self._ending.set()
        self._wake_worker()
        loop = asyncio.get_running_loop()
        try:
            completed = await loop.run_in_executor(None, self._done.wait, timeout_s)
        except asyncio.CancelledError:
            self.close()
            raise
        if not completed:
            self.close()
        if not completed or self._failed or self._closed.is_set():
            return ""
        return self.partial_text()

    def close(self) -> None:
        self._closed.set()
        self._wake_worker()

    def _wake_worker(self) -> None:
        try:
            self._queue.put_nowait(_END)
        except queue.Full:
            # A full queue already has work to wake the decoder. The event
            # makes it exit after draining (finalize) or after this frame (close).
            pass
