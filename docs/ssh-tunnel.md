# Open the voice UI over SSH

The web UI is built and served on the server. Your laptop needs only a browser
and SSH. The Compose deployment binds the gateway and TURN relay to
`127.0.0.1` and creates relay credentials automatically.

An SSH port forward carries the page and session signaling. WebRTC audio
uses a separate media connection. If the devices cannot exchange UDP
directly, use a TURN relay. The following setup keeps coturn on the server's loopback interface and carries its TCP connection through the same SSH
tunnel. This was tested with a laptop browser and a server with an Intel® Xeon® 6 processor.

## Compose deployment

On the server, run `docker compose up` from the cloned repository and wait for
`gateway` to become healthy in `docker compose ps`. Then use the laptop SSH
command below. No separate TURN setup is needed.

## Direct host installation: configure TURN once

For the `make setup` installation, run these commands from the repository
directory. They need Docker and OpenSSL. The password is generated locally, written only to
ignored files, and never placed on the Docker command line.
Stop any earlier TURN service using port 3478 before starting this one.

```bash
TURN_PASS="$(openssl rand -hex 32)"

sed -i 's|^TURN_URL=.*|TURN_URL=turn:127.0.0.1:3478?transport=tcp|' .env
sed -i 's|^TURN_USERNAME=.*|TURN_USERNAME=voice|' .env
sed -i "s|^TURN_PASSWORD=.*|TURN_PASSWORD=$TURN_PASS|" .env

mkdir -p run
cat > run/turnserver.conf <<EOF
listening-ip=127.0.0.1
relay-ip=127.0.0.1
listening-port=3478
min-port=49160
max-port=49200
lt-cred-mech
user=voice:$TURN_PASS
realm=voice-ssh
fingerprint
allow-loopback-peers
no-tls
no-multicast-peers
log-file=stdout
EOF

chmod 600 .env run/turnserver.conf
unset TURN_PASS

docker run -d --name voice-assistant-turn --restart unless-stopped \
    --network host --user 0:0 \
    -v "$PWD/run/turnserver.conf:/etc/coturn/turnserver.conf:ro" \
    coturn/coturn:4.18.0-r0-alpine \
    -c /etc/coturn/turnserver.conf
```

The container runs as root so it can read the mode-600 config file. The
TURN listener and relay are bound to `127.0.0.1`. Check that before use:

```bash
ss -ltn '( sport = :3478 )'
docker logs --tail 20 voice-assistant-turn
```

The listener should show `127.0.0.1:3478`, not `0.0.0.0:3478`. If coturn
reports that it cannot read the config, stop it and fix that before opening
the UI. `allow-loopback-peers` is needed for this private relay because the
gateway and coturn are on the same host. Do not expose port 3478 publicly.

Start the assistant after changing `.env`, or restart it so the gateway
reads the TURN settings:

```bash
make run
make status
curl -fsS http://127.0.0.1:8080/api/ready
```

If the relay was configured earlier and is stopped, use
`docker start voice-assistant-turn` instead of running a second container.

## On the laptop: forward both ports

Keep this command running in a laptop terminal, replacing `user@server` with your SSH destination:

```bash
ssh -N -o ExitOnForwardFailure=yes \
    -L 127.0.0.1:18080:127.0.0.1:8080 \
    -L 127.0.0.1:3478:127.0.0.1:3478 \
    user@server
```

Open <http://127.0.0.1:18080> in the laptop browser, grant microphone
permission, and click the center orb. The `localhost` URL allows browser
microphone access. If the page opens but no transcript appears, first check
that the TURN container is running and that both SSH forwards are present.
Press Ctrl-C in the tunnel terminal when finished.

These local ports must be free before starting the tunnel. The relay
credentials in `.env` and `run/turnserver.conf` must match. If you change
them, restart coturn and the gateway.
