"""Dynamic TURN credentials (Cloudflare Realtime TURN).

Static TURN (TURN_URL/TURN_USERNAME/TURN_PASSWORD) suits self-hosted
coturn. Cloudflare Realtime TURN instead issues *short-lived* credentials
from a TURN key: set CF_TURN_KEY_ID and CF_TURN_API_TOKEN (dashboard:
Realtime -> TURN Server -> Create) and the gateway mints credentials on
demand, caching them until 80% of their TTL so each new session reuses the
same grant. Failure to mint degrades to STUN-only rather than blocking
session creation.
"""

from __future__ import annotations

import logging
import time

import httpx

log = logging.getLogger(__name__)

_API_BASE = "https://rtc.live.cloudflare.com/v1/turn/keys"


def _normalize(payload: dict) -> list[dict]:
    """Accept both API response shapes and return browser-style iceServers."""
    servers = payload.get("iceServers", payload)
    if isinstance(servers, dict):
        servers = [servers]
    out = []
    for s in servers:
        urls = s.get("urls") or []
        if isinstance(urls, str):
            urls = [urls]
        entry: dict = {"urls": urls}
        if s.get("username"):
            entry["username"] = s["username"]
            entry["credential"] = s.get("credential", "")
        out.append(entry)
    return out


class CloudflareTurn:
    def __init__(self, key_id: str, api_token: str, ttl_s: int = 14400) -> None:
        self._key_id = key_id
        self._ttl_s = ttl_s
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_token}"}, timeout=10.0
        )
        self._cached: list[dict] = []
        self._expires_at = 0.0

    async def ice_servers(self) -> list[dict]:
        now = time.monotonic()
        if self._cached and now < self._expires_at:
            return self._cached
        body = {"ttl": self._ttl_s}
        for endpoint in ("credentials/generate-ice-servers", "credentials/generate"):
            try:
                resp = await self._client.post(
                    f"{_API_BASE}/{self._key_id}/{endpoint}", json=body
                )
                if resp.status_code == 404:
                    continue  # older/newer API surface; try the other path
                resp.raise_for_status()
                self._cached = _normalize(resp.json())
                self._expires_at = now + self._ttl_s * 0.8
                log.info("Cloudflare TURN credentials minted (ttl %ss)", self._ttl_s)
                return self._cached
            except httpx.HTTPError as exc:
                log.warning("Cloudflare TURN credential fetch failed: %s", exc)
                break
        return []  # degrade to STUN-only

    async def close(self) -> None:
        await self._client.aclose()
