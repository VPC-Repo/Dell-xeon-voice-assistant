"""Generation epochs: the cancellation backbone.

Every piece of work spawned for a response -- the LLM stream, queued text
segments, TTS jobs, and outbound audio frames -- is stamped with the epoch
current at its creation. Barge-in (or stop/reset) increments the epoch once;
everything stamped with an older epoch is dropped wherever it next surfaces.
That makes cancellation atomic without needing to chase down each queue.
"""

from __future__ import annotations

import asyncio


class EpochController:
    def __init__(self) -> None:
        self._epoch = 0
        self._cancel_event = asyncio.Event()

    @property
    def current(self) -> int:
        return self._epoch

    def is_current(self, epoch: int) -> bool:
        return epoch == self._epoch

    def advance(self) -> int:
        """Invalidate all outstanding work. Returns the new epoch."""
        self._epoch += 1
        # Wake anything waiting on the old epoch, then re-arm for the new one.
        self._cancel_event.set()
        self._cancel_event = asyncio.Event()
        return self._epoch

    def cancel_scope(self, epoch: int) -> EpochScope:
        return EpochScope(self, epoch)


class EpochScope:
    """Bind an awaitable pipeline stage to an epoch.

    `run()` races the wrapped coroutine against epoch invalidation, so a
    stage blocked on a slow backend stops promptly on barge-in instead of
    finishing work nobody will hear.
    """

    def __init__(self, controller: EpochController, epoch: int) -> None:
        self._controller = controller
        self.epoch = epoch

    @property
    def stale(self) -> bool:
        return not self._controller.is_current(self.epoch)

    async def run(self, coro):
        task = asyncio.ensure_future(coro)
        try:
            while True:
                if self.stale:
                    task.cancel()
                    raise EpochCancelled(self.epoch)
                done, _ = await asyncio.wait({task}, timeout=0.05)
                if done:
                    return task.result()
        finally:
            if not task.done():
                task.cancel()


class EpochCancelled(Exception):
    def __init__(self, epoch: int) -> None:
        super().__init__(f"work from epoch {epoch} was cancelled")
        self.epoch = epoch
