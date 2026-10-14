import asyncio

import pytest
from gateway.orchestration.epochs import EpochCancelled, EpochController


def test_advance_invalidates_prior_epoch():
    ctl = EpochController()
    first = ctl.current
    assert ctl.is_current(first)
    ctl.advance()
    assert not ctl.is_current(first)
    assert ctl.is_current(ctl.current)


async def test_scope_completes_when_current():
    ctl = EpochController()
    scope = ctl.cancel_scope(ctl.current)

    async def work():
        await asyncio.sleep(0.01)
        return 42

    assert await scope.run(work()) == 42


async def test_scope_cancels_stale_work():
    ctl = EpochController()
    scope = ctl.cancel_scope(ctl.current)
    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(10)

    task = asyncio.ensure_future(scope.run(slow()))
    await started.wait()
    ctl.advance()
    with pytest.raises(EpochCancelled):
        await asyncio.wait_for(task, timeout=1.0)
