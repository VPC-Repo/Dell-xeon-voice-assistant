#!/usr/bin/env bash
# Start, stop and inspect the assistant. Processes run in the background
# with pids and logs under ./run.
#
#   scripts/run.sh start    inference server, recognition, synthesis, gateway
#   scripts/run.sh stop     stop what start launched (not the LLM container)
#   scripts/run.sh status   health of each part
#
#   MOCK=1 scripts/run.sh start   no models: canned text and tone audio
#
# The language model runs in its own container and is converged rather
# than restarted: one already serving is left alone. STOP_LLM=1 with stop
# shuts it down too.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN="$ROOT/run"
mkdir -p "$RUN"

[ -f .env ] && set -a && . ./.env && set +a

GATEWAY_PORT="${BIND_PORT:-8080}"
GATEWAY_HOST="${BIND_HOST:-127.0.0.1}"
TTS_PORT="${TTS_BIND_PORT:-8082}"
LLM_URL="${LLM_BASE_URL:-http://127.0.0.1:8000}"
PY="$ROOT/.venv/bin/python"

[ -x "$PY" ] || { echo "ERROR: .venv not found. Run 'make setup' first." >&2; exit 1; }

port_open() { (exec 3<>"/dev/tcp/$1/$2") 2>/dev/null && exec 3>&- ; }

wait_http() {  # wait_http URL SECONDS NAME LOGFILE
    local url="$1" secs="$2" name="$3" log="${4:-}"
    for _ in $(seq 1 "$secs"); do
        curl -fs -m 2 "$url" >/dev/null 2>&1 && return 0
        sleep 1
    done
    echo "  $name did not come up within ${secs}s" >&2
    [ -n "$log" ] && [ -f "$log" ] && tail -15 "$log" >&2
    return 1
}

start_one() {  # start_one NAME CMD...
    local name="$1"; shift
    local pidf="$RUN/$name.pid"
    if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
        echo "  $name already running (pid $(cat "$pidf"))"; return 0
    fi
    "$@" >"$RUN/$name.log" 2>&1 &
    echo $! >"$pidf"
    echo "  $name started (pid $!, log run/$name.log)"
}

stop_one() {
    local name="$1" pidf="$RUN/$1.pid"
    [ -f "$pidf" ] || return 0
    local pid; pid="$(cat "$pidf")"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null
        for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
        echo "  $name stopped"
    fi
    rm -f "$pidf"
}

lan_ip() {
    if command -v ipconfig >/dev/null 2>&1; then
        ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null
    else
        hostname -I 2>/dev/null | awk '{print $1}'
    fi
}

cmd_start() {
    if [ "${MOCK:-0}" = "1" ]; then
        export ADAPTER_MODE=mock
        echo "== mock mode: no models, canned replies, tone audio =="
    else
        echo "== language model =="
        if curl -fs -m 3 "$LLM_URL/health" >/dev/null 2>&1 \
           || curl -fs -m 3 "$LLM_URL/v1/models" >/dev/null 2>&1; then
            echo "  ok   serving at $LLM_URL"
        elif [ -x "$ROOT/deployment/sglang/serve.sh" ] && command -v docker >/dev/null 2>&1; then
            echo "  not serving; starting it (first load takes several minutes)"
            bash "$ROOT/deployment/sglang/serve.sh" || {
                echo "  Could not start the inference server. See deployment/sglang/." >&2
                echo "  'MOCK=1 make run' works without it." >&2
                exit 1
            }
        else
            echo "ERROR: nothing serving at $LLM_URL and docker is unavailable." >&2
            echo "  See deployment/sglang/, or run 'MOCK=1 make run'." >&2
            exit 1
        fi

        echo "== synthesis =="
        if [ -f "$ROOT/models/kokoro/kokoro-v1.0.onnx" ]; then
            export TTS_BACKEND="${TTS_BACKEND:-auto}"
            export KOKORO_MODEL_DIR="${KOKORO_MODEL_DIR:-$ROOT/models/kokoro}"
        else
            export TTS_BACKEND=mock
            echo "  warn no Kokoro model: synthesizing tones. Run 'make setup'." >&2
        fi
        export TTS_BIND_PORT="$TTS_PORT"
        start_one tts-server "$ROOT/.venv/bin/tts-server"
        wait_http "http://127.0.0.1:$TTS_PORT/health" 60 "tts-server" "$RUN/tts-server.log" || exit 1
    fi

    echo "== gateway =="
    if port_open 127.0.0.1 "$GATEWAY_PORT" \
       && ! curl -fs -m 3 "http://127.0.0.1:$GATEWAY_PORT/api/health" 2>/dev/null | grep -q '"status"'; then
        echo "ERROR: port $GATEWAY_PORT is held by something that is not this gateway." >&2
        echo "  Set BIND_PORT in .env to a free port." >&2
        exit 1
    fi
    start_one gateway "$ROOT/.venv/bin/voice-gateway"
    wait_http "http://127.0.0.1:$GATEWAY_PORT/api/health" 120 "gateway" "$RUN/gateway.log" || exit 1

    echo
    echo "-------------------------------------------------------------"
    echo "  Open  http://127.0.0.1:$GATEWAY_PORT"
    if [ "$GATEWAY_HOST" = "0.0.0.0" ]; then
        ip="$(lan_ip)"
        [ -n "$ip" ] && echo "  LAN   http://$ip:$GATEWAY_PORT  (text only; browsers need"
        [ -n "$ip" ] && echo "        HTTPS for the microphone -- see docs/deployment.md)"
    fi
    echo "  Stop  make stop"
}

cmd_stop() {
    echo "== stopping =="
    stop_one gateway; stop_one tts-server
    if [ "${STOP_LLM:-0}" = "1" ] && command -v docker >/dev/null 2>&1; then
        docker stop "${CONTAINER_NAME:-sglang-qwen3}" >/dev/null 2>&1 \
            && echo "  inference server stopped"
    fi
}

cmd_status() {
    printf '  %-14s ' "gateway"
    curl -fs -m 3 "http://127.0.0.1:$GATEWAY_PORT/api/health" 2>/dev/null || echo "down"
    echo
    printf '  %-14s ' "tts-server"
    curl -fs -m 3 "http://127.0.0.1:$TTS_PORT/health" 2>/dev/null || echo "down"
    echo
    printf '  %-14s ' "llm"
    if curl -fs -m 3 "$LLM_URL/health" >/dev/null 2>&1 \
       || curl -fs -m 3 "$LLM_URL/v1/models" >/dev/null 2>&1; then
        echo "ok ($LLM_URL)"
    else
        echo "down ($LLM_URL)"
    fi
}

case "${1:-start}" in
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    *) echo "usage: $0 {start|stop|status}" >&2; exit 2 ;;
esac
