#!/usr/bin/env bash
# Build the inference image. Two stages: SGLang's own CPU image, then the
# one-layer fix in ./Dockerfile. Takes 20-40 minutes the first time.
#
#   bash deployment/sglang/build.sh
#
# VER_SGLANG pins the SGLang commit. The default is the revision this
# system was measured on; newer revisions may work but have not been.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VER_SGLANG="${VER_SGLANG:-c04b20dc8}"
SRC="${SGLANG_SRC:-$HERE/.sglang-src}"
IMAGE="${IMAGE:-sglang-cpu:xeon-fixed}"

command -v docker >/dev/null || { echo "ERROR: docker is required." >&2; exit 1; }

if [ ! -d "$SRC/.git" ]; then
    echo "== cloning SGLang =="
    git clone https://github.com/sgl-project/sglang.git "$SRC"
fi
git -C "$SRC" fetch --all --quiet || true
git -C "$SRC" checkout --quiet "$VER_SGLANG"

echo "== building sglang-cpu:xeon (this is the long part) =="
docker build --build-arg "VER_SGLANG=$VER_SGLANG" \
    -f "$SRC/docker/xeon.Dockerfile" -t sglang-cpu:xeon "$SRC"

echo "== adding the tokenizer layer =="
docker build --build-arg BASE=sglang-cpu:xeon -t "$IMAGE" "$HERE"

echo
echo "Built $IMAGE"
echo "Keep a copy -- a docker prune will otherwise cost you the rebuild:"
echo "  docker save $IMAGE | zstd -o sglang-cpu-xeon-fixed.tar.zst"
