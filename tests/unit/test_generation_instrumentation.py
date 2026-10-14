"""Occupancy of the generation gate, and the effective-config line.

Two blind spots motivated both. A run's configuration was recorded only
in notes typed at the time, so no measurement could be reproduced from
its own log. And a turn could report how long it waited for a generation
slot but never how many streams it was decoding alongside -- which is the term that actually explains the wall, since
gen_q is 0 even at ten users while per-stream decode collapses.
"""

from __future__ import annotations

import asyncio

import pytest
from gateway.config import Settings, describe
from gateway.session import GenerationSlots


def settings(**over) -> Settings:
    return Settings(_env_file=None, **over)


def parse(line: str) -> dict[str, str]:
    return dict(p.split("=", 1) for p in line.split(" "))


def test_describe_names_the_capacity_settings():
    got = parse(describe(settings()))
    for key in ("llm_max_tokens", "llm_history_max_tokens", 
                "max_sessions", "max_active_generations", "ack_inject",
                "ack_at_ms"):
        assert key in got, key
    assert got["max_active_generations"] == "4"


def test_describe_reflects_overrides_not_defaults():
    # The point of the line is to catch drift, so it must report what the
    # process actually loaded.
    got = parse(describe(settings(LLM_MAX_TOKENS=192, MAX_ACTIVE_GENERATIONS=2)))
    assert got["llm_max_tokens"] == "192"
    assert got["max_active_generations"] == "2"


def test_describe_carries_no_secrets():
    line = describe(settings(AUTH_TOKEN="hunter2", LLM_API_KEY="sk-secret",  # pragma: allowlist secret
                             CF_TURN_API_TOKEN="cf-secret"))
    assert "hunter2" not in line
    assert "sk-secret" not in line
    assert "cf-secret" not in line


def test_describe_renders_booleans_readably():
    assert "ack_inject=true" in describe(settings(ACK_INJECT=True))
    assert "ack_inject=false" in describe(settings(ACK_INJECT=False))


async def test_slots_report_active_count():
    slots = GenerationSlots(4)
    async with slots as first:
        assert first.active == 1
        assert first.waiting == 0
        async with slots as second:
            assert second.active == 2
    assert slots.active == 0
    assert slots.peak_active == 2


async def test_slots_report_others_waiting_when_full():
    slots = GenerationSlots(1)
    entered = asyncio.Event()
    release = asyncio.Event()
    seen: list[int] = []

    async def holder() -> None:
        async with slots:
            entered.set()
            await release.wait()

    async def latecomer() -> None:
        async with slots as e:
            seen.append(e.waiting)

    task = asyncio.create_task(holder())
    await entered.wait()
    waiters = [asyncio.create_task(latecomer()) for _ in range(2)]
    await asyncio.sleep(0)  # let both reach the gate
    assert slots.waiting == 2
    release.set()
    await asyncio.gather(task, *waiters)
    # The first admitted saw one peer still queued behind it.
    assert max(seen) == 1
    assert slots.waiting == 0
    assert slots.active == 0


async def test_slot_is_released_when_the_body_raises():
    # A generation that dies mid-stream must not leak a slot: leaking one
    # would silently lower capacity for the life of the process.
    slots = GenerationSlots(1)
    with pytest.raises(RuntimeError):
        async with slots:
            raise RuntimeError("stream failed")
    assert slots.active == 0
    async with slots as entry:  # still admits
        assert entry.active == 1
