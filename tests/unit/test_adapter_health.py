"""Readiness must reflect the dependency needed to serve a turn."""

import httpx
import pytest
from gateway.adapters.llm_openai import OpenAICompatLLM
from gateway.adapters.tts_http import HttpStreamingTTS


@pytest.mark.asyncio
@pytest.mark.parametrize("models, available", [(["demo"], True), (["other"], False), ([], False)])
async def test_llm_health_requires_configured_model(models, available):
    adapter = OpenAICompatLLM("http://model.test", model="demo")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://model.test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": [{"id": m} for m in models]})
        ),
    )
    try:
        assert (await adapter.health()).available is available
    finally:
        await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status, available", [(200, True), (404, False), (503, False)])
async def test_tts_health_rejects_error_responses(status, available):
    adapter = HttpStreamingTTS("http://tts.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://tts.test",
        transport=httpx.MockTransport(lambda request: httpx.Response(status)),
    )
    try:
        assert (await adapter.health()).available is available
    finally:
        await adapter.close()
