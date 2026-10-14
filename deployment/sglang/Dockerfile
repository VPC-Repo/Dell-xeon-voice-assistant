# SGLang's CPU image, plus the tokenizer dependencies its FP8 warmup needs.
#
# Build the base from SGLang's own repository first (see build.sh). The
# stock image reaches FP8 warmup and dies there because sentencepiece,
# tiktoken and protobuf are absent from its virtualenv; everything below
# is that one missing layer.
ARG BASE=sglang-cpu:xeon
FROM ${BASE}

RUN /opt/.venv/bin/python -m ensurepip --upgrade \
 && /opt/.venv/bin/python -m pip install --no-cache-dir \
        sentencepiece \
        tiktoken \
        protobuf
