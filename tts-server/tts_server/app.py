"""Streaming TTS HTTP server.

Implements the contract the gateway's HttpStreamingTTS adapter expects:

    POST /synthesize {"text", "voice", "sample_rate", "format": "pcm_s16le"}
        -> chunked raw mono s16le PCM at the requested rate
    GET /health -> backend name, native rate, voice

The model loads once at startup and stays warm. Bind loopback only.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .backends import Backend, build_backend
from .resample import StreamResampler
from .trim import SilenceTrimmer

log = logging.getLogger(__name__)

MAX_TEXT_CHARS = 1000  # segments arrive pre-chunked from the gateway


class SynthesizeBody(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    voice: str = "default"
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    format: str = "pcm_s16le"


def build_app(backend: Backend | None = None) -> FastAPI:
    state: dict[str, Backend] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["backend"] = backend or build_backend()
        log.info(
            "TTS backend %s ready (%d voice(s), default %s); warming up",
            state["backend"].name,
            len(state["backend"].voices()),
            state["backend"].default_voice,
        )
        await state["backend"].warm_up()
        yield

    app = FastAPI(title="tts-server", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.post("/synthesize")
    async def synthesize(body: SynthesizeBody):
        be = state["backend"]
        if body.format != "pcm_s16le":
            raise HTTPException(status_code=400, detail="only pcm_s16le is supported")
        voice = be.resolve_voice(body.voice)
        native_rate = be.sample_rate_for(voice)
        # Normalize segment boundaries: trim the engine's variable lead/tail
        # silence and append one constant tail, so stitched sentences keep an
        # even rhythm. TTS_TRIM_SILENCE=0 disables.
        trim = os.environ.get("TTS_TRIM_SILENCE", "1") != "0"
        tail_ms = int(os.environ.get("TTS_SEGMENT_TAIL_MS", "100"))

        async def generate():
            import time as _time

            t0 = _time.monotonic()
            first_backend_ms = first_out_ms = -1.0
            total_bytes = 0
            resampler = StreamResampler(native_rate, body.sample_rate)
            trimmer = SilenceTrimmer(body.sample_rate) if trim else None
            emitted = False
            async for pcm in be.stream(body.text, voice):
                if first_backend_ms < 0 and pcm:
                    first_backend_ms = (_time.monotonic() - t0) * 1000
                out = resampler.push(pcm)
                if out and trimmer is not None:
                    out = trimmer.push(out)
                if out:
                    if first_out_ms < 0:
                        first_out_ms = (_time.monotonic() - t0) * 1000
                    emitted = True
                    total_bytes += len(out)
                    yield out
            tail = resampler.flush()
            if trimmer is not None:
                tail = trimmer.push(tail) + trimmer.flush()
            if tail:
                if first_out_ms < 0:
                    first_out_ms = (_time.monotonic() - t0) * 1000
                emitted = True
                total_bytes += len(tail)
                yield tail
            if trim and emitted and tail_ms > 0:
                yield bytes(body.sample_rate * 2 * tail_ms // 1000)
            wall = (_time.monotonic() - t0) * 1000
            audio_s = total_bytes / 2 / body.sample_rate
            # Sub-stage audit: backend_first is model inference latency;
            # the first_out gap on top of it is trim/resample cost.
            log.info(
                "synth timing: voice=%s chars=%d backend_first=%.0fms "
                "first_out=%.0fms total=%.0fms audio=%.2fs rtf=%.2f",
                voice, len(body.text), first_backend_ms, first_out_ms,
                wall, audio_s, (wall / 1000 / audio_s) if audio_s else 0,
            )

        return StreamingResponse(generate(), media_type="application/octet-stream")

    @app.get("/voices")
    async def voices():
        be = state.get("backend")
        if be is None:
            raise HTTPException(status_code=503, detail="backend not loaded")
        return {"voices": be.voices(), "default": be.default_voice}

    @app.get("/health")
    async def health():
        be = state.get("backend")
        if be is None:
            raise HTTPException(status_code=503, detail="backend not loaded")
        return {
            "backend": be.name,
            "voice": be.default_voice,
            "voices": len(be.voices()),
            "native_sample_rate": be.sample_rate_for(be.default_voice),
        }

    return app


def main() -> None:
    import os

    import uvicorn

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    uvicorn.run(
        build_app(),
        host=os.environ.get("TTS_BIND_HOST", "127.0.0.1"),
        port=int(os.environ.get("TTS_BIND_PORT", "8082")),
    )


if __name__ == "__main__":
    main()
