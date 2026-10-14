#!/usr/bin/env bash
# One-time setup for everything except the language model, which is a
# separate step (see deployment/sglang/ and the README).
#
# Idempotent and repo-local: virtualenv in .venv, models in ./models,
# logs in ./run. No sudo, nothing installed
# system-wide. Re-run it any time; anything already present is left alone.
#
#   SKIP_FRONTEND=1   API-only setup; no web UI to tunnel
#   SKIP_MODELS=1     skip model downloads (pair with MOCK=1)

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODELS="$ROOT/models"
mkdir -p "$MODELS" "$ROOT/run" "$ROOT/third_party"

OK=(); WARN=()
ok()   { OK+=("$1");   echo "  ok   $1"; }
warn() { WARN+=("$1"); echo "  warn $1" >&2; }

# The browser may run on another device, but this host must build the files
# that the gateway serves. Check before downloading models or installing Python.
node_ready() {
    command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 \
        && node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 20 ? 0 : 1)' \
            >/dev/null 2>&1
}

if [ "${SKIP_FRONTEND:-0}" != "1" ]; then
    if [ -n "${NODE_BIN_DIR:-}" ]; then
        PATH="$NODE_BIN_DIR:$PATH"; export PATH
    fi
    if ! node_ready; then
        for bin in "$HOME"/.local/node-v*/bin \
                   "$HOME"/.nvm/versions/node/v*/bin \
                   "$HOME"/.local/share/mise/installs/node/*/bin \
                   "$HOME"/.asdf/installs/nodejs/*/bin \
                   "$HOME"/.volta/bin; do
            [ -x "$bin/node" ] && [ -x "$bin/npm" ] || continue
            if "$bin/node" -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 20 ? 0 : 1)' \
                    >/dev/null 2>&1; then
                PATH="$bin:$PATH"; export PATH
                echo "  using Node $(node --version) from $bin"
                break
            fi
        done
    fi
    if ! node_ready; then
        echo "ERROR: Node 20+ and npm are needed on this host to build the web UI." >&2
        echo "  Your browser can run on another device through an SSH tunnel." >&2
        echo "  Install Node 20+ with npm, then rerun make setup." >&2
        echo "  For a custom location: NODE_BIN_DIR=/path/to/node/bin make setup" >&2
        exit 1
    fi
fi

fetch() {  # fetch URL DEST MIN_BYTES LABEL
    local url="$1" dest="$2" min="$3" label="$4"
    if [ -f "$dest" ] && [ "$(wc -c < "$dest" | tr -d ' ')" -ge "$min" ]; then
        ok "$label already present"; return 0
    fi
    rm -f "$dest"
    echo "  downloading $label..."
    if curl -fSL --progress-bar -o "$dest" "$url" 2>>"$ROOT/run/download.log" \
            && [ "$(wc -c < "$dest" | tr -d ' ')" -ge "$min" ]; then
        ok "$label downloaded"
    else
        rm -f "$dest"; warn "$label download failed (see run/download.log)"; return 1
    fi
}

echo "== Python =="
PYBIN=""
for c in python3.13 python3.12 python3; do
    command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)' 2>/dev/null; then
        PYBIN="$c"; break
    fi
done
if [ -z "$PYBIN" ]; then
    echo "ERROR: Python 3.12 or newer is required and was not found on PATH." >&2
    echo "  Ubuntu: sudo apt install python3.12 python3.12-venv python3-dev build-essential" >&2
    exit 1
fi
[ -d .venv ] || "$PYBIN" -m venv .venv
.venv/bin/python -m pip install --progress-bar off -q --upgrade 'pip>=26.2' || exit 1
echo "  installing gateway and tts-server (a few minutes the first time)..."
if .venv/bin/pip install --progress-bar off -q -e "./gateway[moonshine]" -e "./tts-server[kokoro]"; then
    ok "gateway + tts-server installed ($("$PYBIN" --version 2>&1))"
else
    echo "ERROR: Python dependency installation failed. Check the pip output above." >&2
    exit 1
fi

if [ "${SKIP_MODELS:-0}" = "1" ]; then
    echo "  skipping models (SKIP_MODELS=1; use MOCK=1 make run)"
else
    echo "== Recognition: streaming =="
    # Moonshine v2 fetches its own quantized weights on first use.
    if .venv/bin/python -c 'import moonshine_voice' 2>/dev/null; then
        ok "moonshine-voice installed (weights download on first run)"
    else
        warn "moonshine-voice missing -- there is no recognizer without it"
    fi

    echo "== Speech synthesis =="
    if .venv/bin/python -c 'import kokoro_onnx' 2>/dev/null; then
        mkdir -p "$MODELS/kokoro"
        k="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
        fetch "$k/kokoro-v1.0.onnx" "$MODELS/kokoro/kokoro-v1.0.onnx" 200000000 "Kokoro voice model"
        fetch "$k/voices-v1.0.bin"  "$MODELS/kokoro/voices-v1.0.bin"   20000000 "Kokoro voice styles"
    else
        warn "kokoro-onnx missing -- synthesis falls back to test tones"
    fi
fi

if [ "${SKIP_FRONTEND:-0}" = "1" ]; then
    echo "== Web UI skipped (SKIP_FRONTEND=1; no interface to tunnel) =="
else
    echo "== Web UI (built here, opened in a browser on any device) =="
    if (cd frontend && npm ci --no-audit --no-fund && npm run build) \
            >"$ROOT/run/frontend-build.log" 2>&1; then
        ok "web UI built (served by the gateway)"
    else
        tail -20 "$ROOT/run/frontend-build.log" >&2
        warn "web UI build failed (run/frontend-build.log); rerun make setup"
    fi
fi

echo "== Configuration =="
if [ -f .env ]; then
    ok ".env already exists (left untouched)"
else
    (umask 077; cp .env.example .env); ok ".env created from .env.example"
fi

echo
echo "-------------------------------------------------------------"
if [ "${#WARN[@]}" -gt 0 ]; then
    echo "Setup incomplete: ${#WARN[@]} issue(s) need attention:"
    for w in "${WARN[@]}"; do echo "  - $w"; done
    echo "Resolve these issues and rerun make setup. Existing downloads are reused."
    exit 1
fi

echo "Setup complete: ${#OK[@]} component(s) ready."
if [ "${SKIP_MODELS:-0}" = "1" ]; then
    echo "Models were intentionally skipped. Test with: MOCK=1 make run"
else
    echo
    echo "Finish the language model on this host (once):"
    echo "  bash deployment/sglang/build.sh      # builds the CPU image; 20-40 min"
    echo "  .venv/bin/python -m pip install -U huggingface_hub"
    echo "  .venv/bin/hf download Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \\"
    echo "      --local-dir \"\$HOME/models/Qwen3-30B-A3B-Instruct-2507-FP8\"  # ~30 GB"
    echo "  make run"
    echo "  make status"
    echo "  Moonshine streaming weights download when the gateway first starts."
fi
if [ "${SKIP_FRONTEND:-0}" != "1" ]; then
    echo
    echo "Open the web UI from another device:"
    echo "  The gateway serves it on this host; no browser or desktop is needed here."
    echo "  For voice, configure the loopback TURN relay: docs/ssh-tunnel.md"
    echo "  Then on your laptop, keep this tunnel running:"
    echo "  ssh -N -L 127.0.0.1:18080:127.0.0.1:8080 \\"
    echo "      -L 127.0.0.1:3478:127.0.0.1:3478 user@server"
    echo "  Open http://127.0.0.1:18080 in the laptop browser."
fi
