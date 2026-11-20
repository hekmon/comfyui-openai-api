import logging
from typing import Any

# ComfyUI prints log messages without the logger name: with many custom node
# packs installed, a message is hard to attribute. Every message of this pack
# starts with this prefix instead (the registry name of the pack).
PREFIX = "[openai-api]"


class _PrefixAdapter(logging.LoggerAdapter):
    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        return f"{PREFIX} {msg}", kwargs


def get_logger(name: str) -> logging.LoggerAdapter:
    """The module logger, its messages prefixed with the pack name."""
    return _PrefixAdapter(logging.getLogger(name), {})
