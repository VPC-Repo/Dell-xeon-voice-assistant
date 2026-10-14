# Based on SGLang docker/xeon.Dockerfile at c04b20dc8.
# Keep the upstream CPU dependency steps pinned to the same revision.
FROM ubuntu:24.04
SHELL ["/bin/bash", "-c"]

ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ARG VER_SGLANG=c04b20dc8829f6c45ce175f389b053795418fb24

RUN apt-get update && \
    apt-get full-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates git curl wget vim gcc g++ make libsqlite3-dev \
    google-perftools libtbb-dev libnuma-dev numactl

WORKDIR /opt
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    source $HOME/.local/bin/env && uv venv --python 3.12

RUN echo -e '[[index]]\nname = "torch"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "torchvision"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "torchaudio"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "triton"\nurl = "https://download.pytorch.org/whl/cpu"' > .venv/uv.toml
ENV UV_CONFIG_FILE=/opt/.venv/uv.toml

WORKDIR /sgl-workspace
RUN source $HOME/.local/bin/env && \
    source /opt/.venv/bin/activate && \
    mkdir sglang && cd sglang && git init && \
    git remote add origin ${SGLANG_REPO} && \
    git fetch --depth 1 origin ${VER_SGLANG} && git checkout FETCH_HEAD && \
    cd python && cp pyproject_cpu.toml pyproject.toml && uv pip install . && \
    cd ../sgl-kernel && cp pyproject_cpu.toml pyproject.toml && uv pip install .

RUN /opt/.venv/bin/python -m ensurepip --upgrade && \
    /opt/.venv/bin/python -m pip install --no-cache-dir \
    sentencepiece tiktoken protobuf

ENV SGLANG_USE_CPU_ENGINE=1
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4:/usr/lib/x86_64-linux-gnu/libtbbmalloc.so:/opt/.venv/lib/libiomp5.so
RUN echo 'source /opt/.venv/bin/activate' >> /root/.bashrc
WORKDIR /sgl-workspace/sglang
