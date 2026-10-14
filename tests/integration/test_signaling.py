"""Signaling API integration tests using in-process fake adapters."""


import time

import pytest
from fastapi.testclient import TestClient
from gateway.app import build_app
from gateway.config import Settings
from gateway.session import SessionManager
from pydantic import ValidationError

from .fakes import FakeLLM, FakeStreamingSTT, FakeTTS


def make_client(**settings_overrides) -> TestClient:
    settings = Settings(_env_file=None, **settings_overrides)
    manager = SessionManager(settings, FakeLLM(["ok. "]), FakeTTS(),
                             streaming_stt=FakeStreamingSTT())
    return TestClient(build_app(settings, manager))


def test_health():
    with make_client() as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_ready_reports_dependency_matrix():
    with make_client() as client:
        resp = client.get("/api/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ready"] is True
        assert set(body["dependencies"]) == {"stt", "llm", "tts"}


def test_session_create_and_delete():
    with make_client() as client:
        resp = client.post("/api/sessions")
        assert resp.status_code == 200
        sid = resp.json()["session_id"]
        assert resp.json()["limits"]["max_utterance_seconds"] > 0
        # Browser builds its RTCPeerConnection from this.
        assert resp.json()["ice_servers"][0]["urls"]
        assert client.delete(f"/api/sessions/{sid}").status_code == 204
        assert client.delete(f"/api/sessions/{sid}").status_code == 404


def test_session_limit_enforced():
    with make_client(MAX_SESSIONS=1) as client:
        assert client.post("/api/sessions").status_code == 200
        assert client.post("/api/sessions").status_code == 503


def test_idle_session_expires_without_a_new_client_request():
    with make_client(MAX_SESSION_SECONDS=1) as client:
        sid = client.post("/api/sessions").json()["session_id"]
        session = client.app.state.manager.get(sid)
        session.created_at -= 2
        time.sleep(0.7)
        assert session.closed
        assert client.app.state.manager.get(sid) is None


def test_cross_origin_session_creation_is_rejected():
    with make_client(MAX_SESSIONS=1) as client:
        assert client.post(
            "/api/sessions", headers={"Origin": "https://untrusted.example"}
        ).status_code == 403
        assert client.post("/api/sessions", headers={"Origin": "http://testserver"}).status_code == 200


def test_configured_origin_can_create_sessions():
    with make_client(ALLOWED_ORIGINS=["https://voice.example"]) as client:
        assert client.post(
            "/api/sessions", headers={"Origin": "https://voice.example"}
        ).status_code == 200


def test_auth_token_required_when_configured():
    with make_client(AUTH_TOKEN="secret123") as client:
        assert client.post("/api/sessions").status_code == 401
        assert client.get("/api/ready").status_code == 401
        assert client.get("/api/metrics").status_code == 401
        resp = client.post(
            "/api/sessions", headers={"Authorization": "Bearer secret123"}
        )
        assert resp.status_code == 200
        headers = {"Authorization": "Bearer secret123"}
        assert client.get("/api/ready", headers=headers).status_code == 200
        assert client.get("/api/metrics", headers=headers).status_code == 200


def test_remote_bind_requires_auth_token():
    with pytest.raises(ValidationError, match="AUTH_TOKEN is required"):
        Settings(_env_file=None, BIND_HOST="0.0.0.0", AUTH_TOKEN="")
    assert Settings(_env_file=None, BIND_HOST="0.0.0.0", AUTH_TOKEN="secret123")
    assert Settings(_env_file=None, BIND_HOST="::1", AUTH_TOKEN="")


@pytest.mark.parametrize("setting", ["MAX_SESSIONS", "MAX_ACTIVE_GENERATIONS", "MAX_SESSION_SECONDS"])
def test_capacity_limits_must_be_positive(setting):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{setting: 0})


def test_offer_requires_known_session():
    with make_client() as client:
        resp = client.post(
            "/api/sessions/nope/offer", json={"sdp": "v=0", "type": "offer"}
        )
        assert resp.status_code == 404


def test_invalid_offer_releases_session_capacity():
    with make_client(MAX_SESSIONS=1, STUN_URLS="") as client:
        sid = client.post("/api/sessions").json()["session_id"]
        response = client.post(
            f"/api/sessions/{sid}/offer", json={"sdp": "v=0", "type": "offer"}
        )
        assert response.status_code == 400
        assert client.app.state.manager.get(sid) is None
        assert client.post("/api/sessions").status_code == 200


def test_metrics_exposition():
    with make_client() as client:
        client.post("/api/sessions")
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        assert "va_active_sessions 1" in resp.text
