"""FastAPI application for signaling, health, metrics, and the browser UI."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .adapters.llm_openai import OpenAICompatLLM
from .adapters.tts_http import HttpStreamingTTS
from .config import Settings, describe, ice_servers, load_settings
from .session import SessionManager
from .turn import CloudflareTurn

log = logging.getLogger(__name__)

_DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class _UIStaticFiles(StaticFiles):
    """StaticFiles with explicit cache policy.

    Without Cache-Control, browsers cache by heuristic (a fraction of the
    file's age) — a phone can then pair a fresh index.html with a weeks-old
    app.js and get a dead UI. HTML must always revalidate; the bundle is
    referenced as app.js?v=<content-hash> (frontend/stamp.mjs), so a stale
    cached copy can never be paired with a newer page.
    """

    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        if path.endswith(".html") or path in ("", "."):
            response.headers["Cache-Control"] = "no-cache"
        else:
            response.headers["Cache-Control"] = "public, max-age=3600"
        return response

# Simple in-memory rate limit for session creation: (window_start, count)
_RATE_WINDOW_S = 60
# Allow normal session churn; operators can set SESSION_CREATES_PER_MIN.
_RATE_MIN_CREATES = 10


class OfferBody(BaseModel):
    sdp: str = Field(min_length=1, max_length=200_000)
    type: str = Field(pattern="^offer$")


def build_app(settings: Settings | None = None, manager: SessionManager | None = None) -> FastAPI:
    settings = settings or load_settings()
    logging.basicConfig(level=settings.log_level.upper())
    # Record the effective settings without credentials.
    log.info("effective config: %s", describe(settings))

    if manager is None and settings.adapter_mode == "mock":
        from .adapters.mock import MockLLM, MockStreamingSTT, MockTTS

        log.warning("ADAPTER_MODE=mock: running with canned STT/LLM and tone TTS")
        manager = SessionManager(
            settings,
            streaming_stt=MockStreamingSTT(),
            llm=MockLLM(),
            tts=MockTTS(sample_rate=settings.tts_sample_rate),
        )
    if manager is None:
        # The recognizer: Moonshine v2 streaming, one single-thread
        # instance per admitted session. It decodes while the person
        # speaks and returns punctuated, cased text, so there is no
        # batch pass at end of speech.
        from .adapters.stt_moonshine_v2 import MoonshineV2StreamingSTT

        streaming_stt = MoonshineV2StreamingSTT(
            settings.stt_moonshine_v2_arch,
            cache_root=settings.stt_moonshine_v2_dir,
            sample_rate=settings.stt_sample_rate,
            # One instance per possible concurrent speaker: a session
            # speaks one utterance at a time, so the pool can never be
            # outnumbered.
            instances=settings.max_sessions,
        )
        manager = SessionManager(
            settings,
            streaming_stt=streaming_stt,
            llm=OpenAICompatLLM(
                settings.llm_base_url,
                settings.llm_model,
                api_key=settings.llm_api_key,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                thinking_mode=settings.llm_thinking_mode,
            ),
            tts=HttpStreamingTTS(
                settings.tts_base_url,
                voice=settings.tts_voice,
                sample_rate=settings.tts_sample_rate,
            ),
        )

    cf_turn: CloudflareTurn | None = None
    if settings.cf_turn_key_id and settings.cf_turn_api_token:
        cf_turn = CloudflareTurn(
            settings.cf_turn_key_id, settings.cf_turn_api_token, settings.cf_turn_ttl_s
        )

    async def session_ice_servers() -> list[dict]:
        servers = ice_servers(settings)
        if cf_turn is not None:
            servers += await cf_turn.ice_servers()
        return servers

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def expire_sessions():
            while True:
                await asyncio.sleep(min(60, settings.max_session_seconds / 2))
                now = time.monotonic()
                for sid, session in list(manager.sessions.items()):
                    if now - session.created_at > settings.max_session_seconds:
                        await manager.close_session(sid)

        expiry_task = asyncio.create_task(expire_sessions())
        try:
            yield
        finally:
            expiry_task.cancel()
            await asyncio.gather(expiry_task, return_exceptions=True)
            await manager.close_all()
            for adapter in (manager.llm, manager.tts):
                await adapter.close()
            if cf_turn is not None:
                await cf_turn.close()

    app = FastAPI(title="voice-gateway", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.manager = manager
    rate_state = {"window": 0.0, "count": 0}

    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    def require_auth(request: Request) -> None:
        """Require the configured shared bearer token."""
        origin = request.headers.get("origin")
        if (
            request.method in {"POST", "DELETE"}
            and origin
            and origin != str(request.base_url).rstrip("/")
            and origin not in settings.allowed_origins
        ):
            raise HTTPException(status_code=403, detail="origin not allowed")
        if not settings.auth_token:
            return
        header = request.headers.get("authorization", "")
        supplied = header.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied, settings.auth_token):
            raise HTTPException(status_code=401, detail="authentication required")

    # ---- signaling ---------------------------------------------------------

    rate_max_creates = settings.session_creates_per_min or max(
        _RATE_MIN_CREATES, settings.max_sessions * 2
    )

    @app.post("/api/sessions", dependencies=[Depends(require_auth)])
    async def create_session():
        now = time.monotonic()
        if now - rate_state["window"] > _RATE_WINDOW_S:
            rate_state["window"] = now
            rate_state["count"] = 0
        rate_state["count"] += 1
        if rate_state["count"] > rate_max_creates:
            raise HTTPException(status_code=429, detail="too many session requests")

        session = manager.create()
        if session is None:
            raise HTTPException(status_code=503, detail="session limit reached")
        try:
            ice = await session_ice_servers()
        except BaseException:
            await manager.close_session(session.id)
            raise
        return {
            "session_id": session.id,
            # The browser builds its RTCPeerConnection from this, so ICE
            # topology is deployment config, not a frontend build concern.
            "ice_servers": ice,
            "limits": {
                "max_session_seconds": settings.max_session_seconds,
                "max_utterance_seconds": settings.max_utterance_seconds,
                "max_text_input_chars": settings.max_text_input_chars,
            },
        }

    @app.post("/api/sessions/{session_id}/offer", dependencies=[Depends(require_auth)])
    async def offer(session_id: str, body: OfferBody):
        session = manager.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        if session.pc is not None:
            raise HTTPException(status_code=409, detail="offer already exchanged")

        rtc_config = RTCConfiguration(
            iceServers=[
                RTCIceServer(
                    urls=s["urls"],
                    username=s.get("username"),
                    credential=s.get("credential"),
                )
                for s in await session_ice_servers()
            ]
        )
        pc = RTCPeerConnection(configuration=rtc_config)
        session.pc = pc
        pc.addTrack(session.out_track)

        @pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                session.start_media(track)

        @pc.on("datachannel")
        def on_datachannel(channel):
            if channel.label == "control":
                session.attach_channel(channel)

        @pc.on("connectionstatechange")
        async def on_state_change():
            if pc.connectionState in ("failed", "closed"):
                await manager.close_session(session_id)

        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except asyncio.CancelledError:
            await manager.close_session(session_id)
            raise
        except Exception as exc:
            await manager.close_session(session_id)
            log.warning("session %s: offer failed (%s)", session_id, type(exc).__name__)
            raise HTTPException(status_code=400, detail="Invalid session offer") from None
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    @app.delete("/api/sessions/{session_id}", dependencies=[Depends(require_auth)])
    async def delete_session(session_id: str):
        if not await manager.close_session(session_id):
            raise HTTPException(status_code=404, detail="unknown session")
        return Response(status_code=204)

    @app.get("/api/voices", dependencies=[Depends(require_auth)])
    async def voices():
        return await manager.tts.list_voices()

    # ---- health and observability -----------------------------------------

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/ready", dependencies=[Depends(require_auth)])
    async def ready():
        checks = await asyncio.gather(
            manager.streaming_stt.health(), manager.llm.health(), manager.tts.health()
        )
        matrix = {c.name: {"available": c.available, "detail": c.detail} for c in checks}
        all_up = all(c.available for c in checks)
        return Response(
            content=json.dumps({"ready": all_up, "dependencies": matrix}),
            media_type="application/json",
            status_code=200 if all_up else 503,
        )

    @app.get("/api/metrics", response_class=PlainTextResponse,
             dependencies=[Depends(require_auth)])
    async def metrics():
        lines = [
            "# TYPE va_active_sessions gauge",
            f"va_active_sessions {len(manager.sessions)}",
            "# TYPE va_rejected_sessions_total counter",
            f"va_rejected_sessions_total {manager.rejected_sessions}",
        ]
        totals = manager.closed_totals
        turns = totals.turns + sum(s.metrics.turns for s in manager.sessions.values())
        barge = totals.barge_ins + sum(s.metrics.barge_ins for s in manager.sessions.values())
        cancels = totals.cancellations + sum(
            s.metrics.cancellations for s in manager.sessions.values()
        )
        errors = totals.errors + sum(s.metrics.errors for s in manager.sessions.values())
        underruns = totals.underruns + sum(
            s.out_track.underruns for s in manager.sessions.values()
        )
        depth = sum(s.out_track.queue_depth for s in manager.sessions.values())
        lines += [
            "# TYPE va_turns_total counter", f"va_turns_total {turns}",
            "# TYPE va_barge_ins_total counter", f"va_barge_ins_total {barge}",
            "# TYPE va_cancellations_total counter", f"va_cancellations_total {cancels}",
            "# TYPE va_errors_total counter", f"va_errors_total {errors}",
            "# TYPE va_underruns_total counter", f"va_underruns_total {underruns}",
            "# TYPE va_outbound_queue_frames gauge", f"va_outbound_queue_frames {depth}",
        ]
        for s in manager.sessions.values():
            if s.metrics.last_eos_to_first_audio_ms is not None:
                lines.append(
                    f'va_eos_to_first_audio_ms{{session="{s.id}"}} '
                    f"{s.metrics.last_eos_to_first_audio_ms:.0f}"
                )
        return "\n".join(lines) + "\n"

    # ---- static UI ---------------------------------------------------------

    frontend_dist = (
        Path(settings.frontend_dist) if settings.frontend_dist else _DEFAULT_FRONTEND_DIST
    )
    if frontend_dist.is_dir():
        app.mount("/", _UIStaticFiles(directory=frontend_dist, html=True), name="ui")

    return app


def main() -> None:
    import uvicorn

    settings = load_settings()
    uvicorn.run(build_app(settings), host=settings.bind_host, port=settings.bind_port)


if __name__ == "__main__":
    main()
