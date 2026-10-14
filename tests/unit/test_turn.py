"""Cloudflare Realtime TURN credential provider."""

import httpx
from gateway.turn import CloudflareTurn, _normalize


def test_normalize_dict_shape():
    payload = {
        "iceServers": {
            "urls": ["stun:stun.cloudflare.com:3478", "turn:turn.cloudflare.com:3478"],
            "username": "u1",
            "credential": "c1",
        }
    }
    out = _normalize(payload)
    assert out == [
        {
            "urls": ["stun:stun.cloudflare.com:3478", "turn:turn.cloudflare.com:3478"],
            "username": "u1",
            "credential": "c1",
        }
    ]


def test_normalize_list_shape_and_string_urls():
    payload = {
        "iceServers": [
            {"urls": "stun:stun.cloudflare.com:3478"},
            {"urls": ["turn:turn.cloudflare.com:3478?transport=udp"], "username": "u", "credential": "c"},
        ]
    }
    out = _normalize(payload)
    assert out[0] == {"urls": ["stun:stun.cloudflare.com:3478"]}
    assert out[1]["username"] == "u"


async def test_fetch_caches_until_ttl(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.headers["authorization"] == "Bearer tok"
        return httpx.Response(
            200,
            json={"iceServers": {"urls": ["turn:turn.cloudflare.com:3478"], "username": "u", "credential": "c"}},
        )

    provider = CloudflareTurn("key123", "tok", ttl_s=100)
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer tok"},
    )
    first = await provider.ice_servers()
    second = await provider.ice_servers()
    assert first == second
    assert first[0]["username"] == "u"
    assert calls["n"] == 1  # cached
    await provider.close()


async def test_fetch_failure_degrades_to_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    provider = CloudflareTurn("key123", "tok")
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await provider.ice_servers() == []
    await provider.close()
