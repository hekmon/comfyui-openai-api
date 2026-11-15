import asyncio
import concurrent.futures
import threading
import weakref
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")

# ComfyUI runs every prompt in a new event loop (asyncio.run) and closes it at
# the end of the prompt, while node outputs (the Client) are cached across
# prompts. An async HTTP client binds its connections to the loop it first
# runs on, so the API clients live on this dedicated loop instead: one for the
# whole process, running in a daemon thread (it never blocks the exit).
_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None


def get_loop() -> asyncio.AbstractEventLoop:
    """The API client event loop, started on first use."""
    global _loop
    with _lock:
        if _loop is None:
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="OpenAI API client loop", daemon=True).start()
            _loop = loop
        return _loop


def submit(coro: Coroutine[Any, Any, T]) -> "asyncio.Future[T]":
    """Run a coroutine on the API client loop. The returned future must be
    awaited from the calling event loop (the ComfyUI prompt one): it gives the
    coroutine result or exception, and cancelling it cancels the coroutine."""
    return asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, get_loop()))


def _close_on_loop(http_client: Any) -> None:
    # Called from whatever thread drops the last reference to the API client
    loop = _loop
    if loop is None or loop.is_closed():
        return
    future: concurrent.futures.Future[None] = asyncio.run_coroutine_threadsafe(http_client.aclose(), loop)
    # Nobody waits for it: retrieve any error so that it is not logged as an
    # unretrieved exception, a failure to close an unused client is harmless
    future.add_done_callback(lambda f: f.cancelled() or f.exception())


def close_when_collected(client: Any) -> None:
    """Close the HTTP client (its connections) of an AsyncOpenAI client on the
    API client loop once ComfyUI drops it from its cache and it is garbage
    collected. Without this, the SDK would close it from the event loop running
    at that time, if any: often the ComfyUI prompt one, not its own.

    The finalizer only references the HTTP client, never the AsyncOpenAI
    object, which would otherwise be kept alive forever. Not run at exit: the
    loop thread is a daemon one, its sockets are closed with the process."""
    http_client = getattr(client, "_client", None)  # the SDK owned httpx client
    if http_client is None or not hasattr(http_client, "aclose"):
        return
    weakref.finalize(client, _close_on_loop, http_client).atexit = False
