"""OpenAI-compatible streaming chat adapter.

The target is SGLang serving Qwen3-30B-A3B-Instruct-2507-FP8 on CPU with
the intel_amx attention backend (deployment/sglang/). Nothing here is
specific to it, though: the adapter speaks plain /v1/chat/completions with
`stream: true`, so any runtime exposing that will work -- point
LLM_BASE_URL at it and set LLM_MODEL to the served model name.
Runtime-specific quirks (e.g. how thinking mode is disabled) stay inside
this file.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from .base import AdapterHealth, ChatMessage, LLMAdapter, LLMDelta

log = logging.getLogger(__name__)


class OpenAICompatLLM(LLMAdapter):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        max_tokens: int = 768,
        temperature: float = 0.7,
        thinking_mode: str = "disabled",
        timeout_s: float = 120.0,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._thinking_mode = thinking_mode
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s
        )

    def _request_body(self, messages: list[ChatMessage]) -> dict:
        body: dict = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if self._thinking_mode == "disabled":
            # Hybrid-thinking Qwen3 checkpoints: SGLang and vLLM accept this
            # via chat_template_kwargs. Instruct-2507 has no thinking mode at
            # all -- use LLM_THINKING_MODE=none there so nothing is sent.
            # Sanitization downstream still strips <think> blocks either way.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[LLMDelta]:
        async with self._client.stream(
            "POST", "/v1/chat/completions", json=self._request_body(messages)
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    yield LLMDelta(done=True)
                    return
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    log.warning("unparseable SSE line from LLM: %r", payload[:200])
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                finish = choice.get("finish_reason")
                if text or finish:
                    yield LLMDelta(text=text, done=finish is not None, finish_reason=finish)
        yield LLMDelta(done=True)

    async def health(self) -> AdapterHealth:
        try:
            resp = await self._client.get("/v1/models", timeout=5.0)
            resp.raise_for_status()
            ids = [m.get("id") for m in resp.json().get("data", [])]
            return AdapterHealth(
                name="llm",
                available=self._model in ids,
                detail=f"configured={self._model}; models={ids}",
                extra={"model_listed": self._model in ids},
            )
        except Exception as exc:  # noqa: BLE001 - health probes report, not raise
            return AdapterHealth(name="llm", available=False, detail=str(exc))

    async def close(self) -> None:
        await self._client.aclose()
