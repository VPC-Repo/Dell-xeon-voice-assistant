#!/usr/bin/env bash
# Serve Qwen3-30B-A3B-Instruct-2507-FP8 on CPU with Intel AMX.
#
#   bash deployment/sglang/serve.sh
#
# Converges rather than restarts: an instance already serving is left
# alone, one still loading is waited for, and a wedged container with the
# configured name is replaced. FORCE=1 relaunches unconditionally.
set -uo pipefail

IMAGE="${IMAGE:-sglang-cpu:xeon-fixed}"
NAME="${CONTAINER_NAME:-sglang-qwen3}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
MODEL="${MODEL:-Qwen3-30B-A3B-Instruct-2507-FP8}"
SERVED_NAME="${SERVED_NAME:-qwen3_30b_a3b}"
PORT="${PORT:-8000}"
LOAD_GRACE="${LOAD_GRACE:-900}"   # seconds a fresh container may spend loading
FORCE="${FORCE:-0}"

serving() { curl -fs -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

wait_serving() {
    local deadline=$((SECONDS + LOAD_GRACE))
    while [ "$SECONDS" -lt "$deadline" ]; do
        serving && return 0
        sleep 5
    done
    return 1
}

if [ "$FORCE" != "1" ] && serving; then
    echo "Already serving on port $PORT."
    exit 0
fi

if [ "$FORCE" != "1" ] && docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    started=$(docker inspect -f '{{.State.StartedAt}}' "$NAME" 2>/dev/null)
    age=$(( $(date +%s) - $(date -d "$started" +%s 2>/dev/null || echo 0) ))
    if [ "$age" -lt "$LOAD_GRACE" ]; then
        echo "Container started ${age}s ago; the model is probably still loading."
        echo "Watching until it serves (Ctrl-C is safe):"
        wait_serving && { echo "  serving."; exit 0; }
    fi
    echo "Container is ${age}s old and not serving -- replacing it."
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image $IMAGE not found." >&2
    echo "  Build it:  bash deployment/sglang/build.sh" >&2
    exit 1
fi
if [ ! -d "$MODELS_DIR/$MODEL" ]; then
    echo "ERROR: weights not found at $MODELS_DIR/$MODEL" >&2
    echo "  Download (~30 GB):" >&2
    echo "    huggingface-cli download Qwen/$MODEL --local-dir $MODELS_DIR/$MODEL" >&2
    exit 1
fi

# Replace only the container managed by this script.
docker rm -f "$NAME" >/dev/null 2>&1
sleep 2

echo "Starting $NAME ..."
docker run -d --name "$NAME" \
    --publish "127.0.0.1:$PORT:$PORT" \
    --shm-size 32g \
    -v "$MODELS_DIR:/models" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$IMAGE" \
    /opt/.venv/bin/python3 -m sglang.launch_server \
        --model-path "/models/$MODEL" \
        --served-model-name "$SERVED_NAME" \
        --device cpu \
        --quantization fp8 \
        --attention-backend intel_amx \
        --max-running-requests 64 \
        --context-length 8192 \
        --chunked-prefill-size 4096 \
        --max-total-tokens 16384 \
        --mem-fraction-static 0.85 \
        --disable-overlap-schedule \
        --trust-remote-code \
        --host 0.0.0.0 --port "$PORT" >/dev/null

echo "Loading 30 GB of weights; first start takes several minutes."
wait_serving && { echo "Serving on port $PORT."; exit 0; }
echo "ERROR: did not come up within ${LOAD_GRACE}s. Logs:" >&2
docker logs --tail 30 "$NAME" >&2
exit 1
