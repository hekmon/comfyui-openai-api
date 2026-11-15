import asyncio
from typing import Any, Awaitable, TypeVar

import comfy.model_management

T = TypeVar("T")

# How often a running request checks the ComfyUI interrupt flag, in seconds
_POLL_INTERVAL = 0.1


async def run_interruptible(awaitable: Awaitable[T]) -> T:
    """Await a long operation (an HTTP request) while watching the ComfyUI
    interrupt flag, set by the Cancel button and POST /interrupt.

    ComfyUI never cancels a running async node: it only sets this flag and
    checks it before starting the next node, so a request would keep running
    until it ends (or times out). This cancels it as soon as the flag is set
    and raises the ComfyUI interrupt exception, which ends the prompt the same
    way as an interrupted sampler.

    The awaitable can be a coroutine (run on the current loop) or a future
    from aioloop.submit (cancelling it cancels the request on its loop)."""
    task: asyncio.Future[Any] = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=_POLL_INTERVAL)
            if done:
                return task.result()
            if comfy.model_management.processing_interrupted():
                task.cancel()
                # Let the request clean up (close its connection). This does
                # not raise the task exception, only our own cancellation. A
                # request running on the API client loop (aioloop.submit)
                # cleans up there: this returns at once.
                await asyncio.wait({task})
                # Clears the flag and raises, like the ComfyUI samplers do.
                # Another node running in parallel may have cleared it first:
                # raise anyway, this node was interrupted too.
                comfy.model_management.throw_exception_if_processing_interrupted()
                raise comfy.model_management.InterruptProcessingException()
    finally:
        # Also covers our own cancellation (e.g. the end of the prompt event
        # loop after another node failed): never leave the request running
        if not task.done():
            task.cancel()
