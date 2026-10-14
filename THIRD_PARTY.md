# Third-party components

This repository is Apache 2.0. It does not include model weights or
third-party source code. Models and runtimes installed separately retain
their own license terms. Check those terms before redistributing an image
or appliance.

## Models

| Component | Role | License |
|---|---|---|
| [Qwen3-30B-A3B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8) | answer generation | Apache 2.0 |
| [Moonshine v2](https://github.com/moonshine-ai/moonshine) (`medium-streaming`) | streaming recognition | MIT |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | speech synthesis | Apache 2.0 |

Moonshine's weights download from `download.moonshine.ai` on first use.
Moonshine lists its streaming models as MIT. Some older, non-streaming
non-English models have different terms. Check the license for any model
you substitute.

## Runtimes and libraries

| Component | Role | License |
|---|---|---|
| [SGLang](https://github.com/sgl-project/sglang) | inference server | Apache 2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) (CPU wheels) | inside the inference image | BSD 3-Clause |
| [moonshine-voice](https://github.com/moonshine-ai/moonshine) | streaming recognition runtime | MIT |
| [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | synthesis runtime | MIT |
| [onnxruntime](https://github.com/microsoft/onnxruntime) | inference | MIT |
| [aiortc](https://github.com/aiortc/aiortc) | WebRTC | BSD 3-Clause |
| [FastAPI](https://github.com/fastapi/fastapi) / [uvicorn](https://github.com/encode/uvicorn) | HTTP | MIT / BSD 3-Clause |
| [webrtcvad-wheels](https://github.com/daanzu/py-webrtcvad-wheels) | voice activity detection | MIT, wrapping Google's WebRTC (BSD 3-Clause) |

The inference image pulls a full Python and PyTorch stack of its own.
`docker history voice-assistant-sglang:local` shows what it contains.

## Verifying this list

Licenses change between releases. To check what you actually installed:

```bash
.venv/bin/pip list --format=freeze | cut -d= -f1 | \
  xargs -n1 .venv/bin/pip show 2>/dev/null | grep -E '^(Name|License):'
```
