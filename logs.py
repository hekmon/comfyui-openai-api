import logging
from typing import Any

from . import aioloop

# ComfyUI prints log messages without the logger name: with many custom node
# packs installed, a message is hard to attribute. Every message of this pack
# starts with this prefix instead (the registry name of the pack).
PREFIX = "[openai-api]"

# The library loggers logging at INFO or above for the requests of this pack:
# httpx logs every request ('HTTP Request: POST ... "HTTP/1.1 200 OK"') and the
# OpenAI SDK every retry ("Retrying request ..."). A logger filter only sees
# the records created on that exact logger, not on its children: these are
# the names the records are created on, not their parents.
LIBRARY_LOGGERS = ("httpx", "openai._base_client")


class _PrefixAdapter(logging.LoggerAdapter):
    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        return f"{PREFIX} {msg}", kwargs


def get_logger(name: str) -> logging.LoggerAdapter:
    """The module logger, its messages prefixed with the pack name."""
    return _PrefixAdapter(logging.getLogger(name), {})


class _LoopThreadPrefixFilter(logging.Filter):
    """Prefixes the library log messages emitted for the requests of this pack,
    then tags them with the library name ("[openai-api] [httpx] ..."): the
    reader knows the pack did not write them itself.

    Other node packs and ComfyUI itself may use the same libraries: only the
    records created on the API client loop thread, where every request of this
    pack runs, are prefixed. Never drops a record."""

    # Marks the filter across module reloads, whose class is a new one
    openai_api_prefix = True

    def filter(self, record: logging.LogRecord) -> bool:
        thread_id = aioloop.loop_thread_id()
        if thread_id is not None and record.thread == thread_id:
            msg = str(record.msg)
            if not msg.startswith(PREFIX):
                # The top-level package: "openai", not "openai._base_client"
                library = record.name.split(".")[0]
                # Only the format string: the arguments stay as they are, still
                # formatted lazily, only if a handler prints the record
                record.msg = f"{PREFIX} [{library}] {msg}"
        return True


def _install_library_filter() -> None:
    # Idempotent: a reload of this module replaces the filter of the previous
    # one instead of adding a second one
    prefix_filter = _LoopThreadPrefixFilter()
    for name in LIBRARY_LOGGERS:
        library_logger = logging.getLogger(name)
        for old in [f for f in library_logger.filters if getattr(f, "openai_api_prefix", False)]:
            library_logger.removeFilter(old)
        library_logger.addFilter(prefix_filter)


_install_library_filter()
