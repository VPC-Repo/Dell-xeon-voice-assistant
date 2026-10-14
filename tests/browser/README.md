# Browser smoke test

The automated smoke test runs the gateway with mock adapters and uses a
Chromium fake microphone. No model files are needed. From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ./gateway
npm ci --prefix frontend
npm run build --prefix frontend
npm exec --prefix frontend -- playwright install chromium
GATEWAY_PYTHON=.venv/bin/python npm run test:browser --prefix frontend
```

The test verifies a WebRTC connection, a recognized utterance, streamed
mock text, and audio playback. It also runs in CI. The checklist below
still needs a person and a real microphone and speaker.

## Manual smoke checklist

1. Open the HTTPS URL in a fresh browser profile. No install prompts appear.
2. Click the center orb; approve the microphone permission.
3. Enable **Show live captions** in settings. Speak a question. Confirm
   that text appears and playback begins before the answer finishes streaming.
4. Interrupt mid-answer by speaking. Confirm playback stops and the
   assistant begins listening again.
5. Repeat with laptop speakers (not headphones) and confirm the assistant's
   own voice does not trigger false turns.
6. Click **Mute**, speak, confirm nothing happens. Unmute and continue.
7. Reload the page mid-conversation; reconnect succeeds as a fresh session.
