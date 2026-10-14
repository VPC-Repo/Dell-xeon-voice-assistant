# Running it for other people

The quick start runs on `127.0.0.1`, where browsers grant microphone
access. Serving other devices needs two things the local
case gives you for free: a trusted certificate, and a path for the audio.

## HTTPS

Browsers expose `getUserMedia` only on `localhost` or a secure origin.
Over plain HTTP from another machine the page loads, but microphone
capture will not work.

Self-signed certificates work only after each client trusts the issuing
certificate authority. A publicly trusted certificate avoids that step.

Two approaches work:

**A reverse proxy with ACME.** Caddy or Traefik in front of the gateway,
obtaining a certificate automatically. If the host is not reachable from
the internet on 80/443, use a DNS-01 challenge and split DNS pointing the
name at the host's private address.

```
voice.example.com {
	encode gzip
	reverse_proxy 127.0.0.1:8080

	# Keep metrics off the public route even when AUTH_TOKEN is set.
	@metrics path /api/metrics*
	respond @metrics 403

	header {
		Strict-Transport-Security "max-age=31536000"
		X-Content-Type-Options "nosniff"
		Referrer-Policy "no-referrer"
	}
}
```

**A tunnel.** Cloudflare Tunnel, ngrok or similar terminate TLS for you
and need no inbound ports. An HTTP tunnel carries the page and signaling,
but WebRTC media needs its own path. For one laptop connecting over SSH,
[forward a loopback TURN relay as well](ssh-tunnel.md). Other tunnels still
need direct media connectivity or a reachable TURN service.

## Getting the audio through

Audio is a direct peer connection between the browser and the host.

- **Same LAN:** works on host candidates alone, nothing to configure.
- **Across networks:** needs STUN to find a route. One is configured by
  default (`STUN_URLS`).
- **Strict or symmetric NAT:** needs a TURN relay, or the connection
  fails. Set `TURN_URL` / `TURN_USERNAME` / `TURN_PASSWORD` for a
  self-hosted relay such as coturn, or `CF_TURN_KEY_ID` /
  `CF_TURN_API_TOKEN` for Cloudflare Realtime TURN, where the gateway
  mints short-lived credentials per session.

Test from a separate network, such as a phone on cellular. If the page
loads but no audio or transcript arrives, inspect the browser's ICE
connection state and configure TURN if direct connectivity fails.

## Before you share the URL

A tunnel or a public DNS name makes the gateway reachable by anyone who
finds it, and every session consumes generation capacity.

- Set `AUTH_TOKEN` to a long random value (for example, generate one with
  `openssl rand -hex 32`). The client prompts once and remembers it for
  that browser origin. Keep `.env` out of Git.
- Keep `BIND_HOST=127.0.0.1` when a reverse proxy runs on the same host.
  A non-loopback bind requires `AUTH_TOKEN` at startup.
- For stronger access control, put an identity-aware proxy in front, such as Cloudflare Access, `forward_auth` or your own SSO.
- Keep `MAX_SESSIONS` and `MAX_ACTIVE_GENERATIONS` at values the host can
  actually sustain. They are the backstop against a busy day and against
  someone pointing a script at you.
- Block `/api/metrics` at the public reverse proxy.

## Running it as a service

Nothing here depends on the repo scripts. The two processes are:

```
.venv/bin/tts-server      # synthesis, port 8082
.venv/bin/voice-gateway   # gateway, port 8080
```

The gateway reads `.env` from the working directory. The TTS server reads
environment variables. `make run` exports `.env` for both. Both log to stdout. Wrap them
in whatever your host uses, such as systemd units, a container or a supervisor. The
gateway needs the synthesis server. Neither needs root.

## Tuning for your host

The capacity settings in `.env.example` assume a many-core Intel® Xeon® 6 processor with Intel® AMX.

**Recognition competes with generation.** A batch recognizer does a burst
of heavy work right after someone stops speaking, exactly when the model
is starting to generate. On a bandwidth-bound host that burst measurably
slows generation. Streaming recognition spreads the same work thinly
across the person's speech and the interference largely disappears. On a
quiet host the two approaches look nearly identical. Under load they do
not. This is why recognition streams.

**More parallelism stops helping, then hurts.** Extra recognition
workers, extra synthesis threads and extra generation slots all raise
throughput up to the point where memory bandwidth saturates, and lower it
after. The useful limit is a property of the host, and the only way to
find it is to measure on that host: raise `MAX_ACTIVE_GENERATIONS` one
step at a time and watch whether the median answer time stays flat or
starts climbing for everyone at once. When it climbs, you have passed the
limit, and adding slots past that point makes every speaker wait.
