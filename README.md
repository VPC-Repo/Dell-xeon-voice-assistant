# CPU voice assistant

A browser voice assistant demo for a many-core Intel Xeon host. It runs
speech recognition, response generation, and speech synthesis on CPU, with
WebRTC carrying audio between the browser and server. Responses play as
short speech segments while the model is still generating. Speaking over
the assistant interrupts its reply.

The supplied inference deployment targets **x86-64 Linux with Intel AMX**,
such as Intel Xeon 6. Other CPUs and model servers need their own deployment
and capacity settings. This is a demo, with a shared access token rather
than individual user accounts.

## Components

| Component | Implementation |
|---|---|
| Browser | TypeScript client with microphone capture and audio playback |
| Gateway | FastAPI and aiortc; turn detection, interruption, and session management |
| Recognition | Moonshine v2 `medium-streaming` |
| Generation | Qwen3-30B-A3B-Instruct-2507-FP8 on SGLang with `intel_amx` attention |
| Synthesis | Kokoro-82M |

```text
microphone → WebRTC → voice detection → recognition → language model
                                                        ↓
speaker ← WebRTC ← audio buffering ← synthesis ← speech segments
```

Recognition, generation, and synthesis run on the configured host services.
Installation and model downloads require internet access. The default STUN
service receives network metadata; a configured TURN relay carries audio.

## Quick start with Docker Compose

Install Docker Engine and the Compose plugin on the Linux server. Your
account must be able to run Docker; host Python and Node installations are
not required.

```bash
git clone https://github.com/VPC-Repo/Dell-xeon-voice-assistant.git
cd Dell-xeon-voice-assistant
docker compose up --build
```

The first run builds the application and SGLang images, downloads Qwen
weights (about 30 GB) and Kokoro, and loads the model. Moonshine downloads
its weights when the gateway starts. Allow substantial time and disk space.
Models and generated TURN credentials persist in Docker volumes.

```bash
docker compose ps                 # wait for gateway to become healthy
docker compose logs -f gateway    # inspect startup and runtime logs
docker compose down              # stop the deployment
```

Subsequent starts use `docker compose up`. After changing source files, use
`docker compose up --build`. `docker compose down -v` also deletes downloaded
models and relay credentials, so the next start downloads them again.

The UI and TURN relay bind to the server's loopback address. On your laptop,
keep this SSH tunnel running, replacing `user@server` with your destination:

```bash
ssh -N -o ExitOnForwardFailure=yes \
    -L 127.0.0.1:18080:127.0.0.1:8080 \
    -L 127.0.0.1:3478:127.0.0.1:3478 \
    user@server
```

Open <http://127.0.0.1:18080>, grant microphone permission, and click the
center orb. Both port forwards are required for this setup. See
[SSH access and TURN setup](docs/ssh-tunnel.md) for troubleshooting and the
direct host installation's relay configuration.

## Direct host installation

Install Python 3.12+, Node 20+ with npm, cmake, a C++ compiler, and Docker.
From the repository directory:

```bash
make setup
bash deployment/sglang/build.sh
.venv/bin/python -m pip install -U huggingface_hub
.venv/bin/hf download Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --local-dir "$HOME/models/Qwen3-30B-A3B-Instruct-2507-FP8"
make run
```

Setup creates `.venv`, copies `.env.example` to `.env`, downloads Kokoro,
and builds the UI. Moonshine downloads on first use. For a custom Node
location, use `NODE_BIN_DIR=/path/to/node/bin make setup`.

The UI is at <http://127.0.0.1:8080> on the server. To use it from a laptop,
configure the relay using [the SSH guide](docs/ssh-tunnel.md).

```bash
make status                     # service health
make stop                       # stop gateway and synthesis
STOP_LLM=1 make stop             # also stop the managed inference container
```

The launcher reuses an inference server that is already serving. Direct
host and Compose installations use the same default ports; run one at a time.

## Try without models

To test microphone capture, WebRTC, interruption, and playback with canned
text and tones:

```bash
SKIP_MODELS=1 make setup
MOCK=1 make run
```

Setup still installs Python and frontend dependencies. For access from
another device, configure the relay and SSH forwards described above.

## Configuration

Direct host installs read `.env`. Compose service settings are in
`compose.yaml`; add or change gateway environment entries there and restart
with `docker compose up`.

| Setting | Purpose |
|---|---|
| `PROMPT_CONTEXT_FILE` | Text file with a persona and facts |
| `MAX_ACTIVE_GENERATIONS` | Maximum answers generating at once |
| `MAX_SESSIONS` | Concurrent sessions and recognition pool size |
| `VAD_END_SILENCE_MS` | Silence needed to confirm the end of a turn |
| `ACK_AT_MS` | Delay before a prepared acknowledgment plays |

For the included grocery store example, set
`PROMPT_CONTEXT_FILE=prompts/grocery-store.txt` in `.env`. In Compose, add
`PROMPT_CONTEXT_FILE: /app/prompts/grocery-store.txt` to the gateway's
`environment`; `prompts/` is already mounted.

See [configuration](docs/configuration.md) for tuning and
[deployment](docs/deployment.md) for HTTPS and remote media connectivity.
Browsers require localhost or trusted HTTPS for microphone access.

## Development

| Directory | Contents |
|---|---|
| `gateway/` | Session pipeline, model adapters, and WebRTC media |
| `tts-server/` | Streaming synthesis service |
| `frontend/` | Browser UI |
| `deployment/sglang/` | CPU inference images and launcher |
| `tests/` | Python, browser, and container checks |

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e './gateway[dev]' -e ./tts-server
make test
make lint
npm ci --prefix frontend
npm run typecheck --prefix frontend
npm run build --prefix frontend
```

Python tests use fake adapters and require no model downloads. The
[Chromium smoke test](tests/browser/README.md) checks the browser audio path.
CI runs these checks and an application container smoke test.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the full validation commands.

## License

[Apache 2.0](LICENSE). Downloaded models and runtimes retain their own
licenses; see [THIRD_PARTY.md](THIRD_PARTY.md). For private issue reporting,
see [SECURITY.md](SECURITY.md).
