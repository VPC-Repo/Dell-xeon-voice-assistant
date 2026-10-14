# Contributing

Use Python 3.12 or newer and Node 20 or newer. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e './gateway[dev]' -e ./tts-server
npm ci --prefix frontend
make test
make lint
make secrets
npm run typecheck --prefix frontend
npm run build --prefix frontend
bash -n scripts/setup.sh scripts/run.sh deployment/sglang/build.sh deployment/sglang/serve.sh
docker compose config -q
```

The Python tests use fake adapters and do not download models. For the
browser test, install Chromium and follow [the browser test guide](tests/browser/README.md).
To try the UI manually, build it, then run `MOCK=1 make run` and open
<http://127.0.0.1:8080>. Stop the gateway with `make stop`.

CI also builds the application image and checks its CLI services, static UI
and mock synthesis without models or external network access. Run the same check on an x86-64 Linux Docker host:

```bash
docker build -t voice-assistant-check .
docker run --rm --network none \
  -v "$PWD/tests/container/smoke.py:/tmp/smoke.py:ro" \
  voice-assistant-check python /tmp/smoke.py
```

Keep changes focused on the CPU voice pipeline, deployment, and browser
client. Add regression coverage for changes to cancellation, bounded queues,
session cleanup, or signaling. Describe the trigger, resulting behavior,
and checks in your pull request.

Test model or capacity changes on the target Xeon. Include the hardware,
model revision, effective settings, and latency measurements when reporting
performance. The default settings are a starting point for that hardware.

Keep credentials, recordings, downloaded weights, and host-specific files
out of commits. Check staged changes before pushing. Report vulnerabilities
using [SECURITY.md](SECURITY.md).
