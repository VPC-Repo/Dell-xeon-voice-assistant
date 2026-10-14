from gateway.config import Settings, ice_servers


def test_default_is_stun_only():
    servers = ice_servers(Settings(_env_file=None))
    assert servers == [{"urls": ["stun:stun.cloudflare.com:3478"]}]


def test_empty_stun_means_lan_only():
    servers = ice_servers(Settings(_env_file=None, STUN_URLS=""))
    assert servers == []


def test_turn_appended_with_credentials():
    servers = ice_servers(
        Settings(
            _env_file=None,
            STUN_URLS="stun:a.example:3478, stun:b.example:3478",
            TURN_URL="turn:relay.example:3478?transport=udp",
            TURN_USERNAME="demo",
            TURN_PASSWORD="secret",  # pragma: allowlist secret
        )
    )
    assert servers[0]["urls"] == ["stun:a.example:3478", "stun:b.example:3478"]
    assert servers[1] == {
        "urls": ["turn:relay.example:3478?transport=udp"],
        "username": "demo",
        "credential": "secret",
    }
