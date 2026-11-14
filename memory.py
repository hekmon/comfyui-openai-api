import copy
import threading
from typing import Any

from comfy_api.latest import io


# In-process memory shared by every prompt: a helper to carry a value (an
# analysis, a conversation history...) from one queued run to the next. It is
# kept in RAM only and lost when ComfyUI restarts.
class _MemoryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, Any] = {}
        # Per key modification counter, bumped on every save or wipe. Never
        # reset nor removed (even on wipe): a counter going back to a previous
        # value could make ComfyUI serve a stale cached Load Memory output.
        self._versions: dict[str, int] = {}

    def save(self, key: str, value: Any) -> None:
        stored = copy.deepcopy(value)
        with self._lock:
            self._values[key] = stored
            self._versions[key] = self._versions.get(key, 0) + 1

    def wipe(self, key: str) -> None:
        with self._lock:
            self._values.pop(key, None)
            self._versions[key] = self._versions.get(key, 0) + 1

    def load(self, key: str) -> tuple[bool, Any]:
        with self._lock:
            if key not in self._values:
                return False, None
            stored = self._values[key]
        # Stored values are never mutated in place (save replaces them), so
        # the copy can be made outside of the lock
        return True, copy.deepcopy(stored)

    def version(self, key: str) -> int:
        with self._lock:
            return self._versions.get(key, 0)


MEMORY = _MemoryStore()


def normalize_key(key: str | None) -> str:
    return key.strip() if isinstance(key, str) else ""


class SaveMemory(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_SaveMemory",
            display_name="OpenAI API - Save Memory",
            category="OpenAI API/Memory",
            description="Stores a copy of any value (text, history...) in memory under a key, so a Load Memory node can return it on the next queued run. The memory is kept in RAM only, shared by all prompts and lost when ComfyUI restarts. The value is passed through unchanged.",
            is_output_node=True,
            inputs=[
                io.String.Input(
                    id="key",
                    display_name="Key",
                    tooltip="The name under which the value is stored. Leading and trailing spaces are ignored.",
                    default="memory",
                ),
                io.AnyType.Input(
                    id="value",
                    display_name="Value",
                    tooltip="The value to store (any type)",
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    id="value_out",
                    display_name="Value",
                    tooltip="The input value, passed through unchanged",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, key: str | None) -> bool | str:
        # key is None when it comes from another node: checked at execution
        if key is not None and normalize_key(key) == "":
            return "key must not be empty"
        return True

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> float:
        # Always execute: after a reset of the key, unchanged (cached) upstream
        # nodes must still store their value again
        return float("nan")

    @classmethod
    def execute(cls, key: str, value: Any) -> io.NodeOutput:
        normalized_key = normalize_key(key)
        if normalized_key == "":
            raise ValueError("Save Memory: key must not be empty")
        try:
            MEMORY.save(normalized_key, value)
        except Exception as e:
            raise ValueError(
                f"Save Memory: this value can not be copied into memory ({type(e).__name__}: {e})") from e
        return io.NodeOutput(value)


class LoadMemory(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_LoadMemory",
            display_name="OpenAI API - Load Memory",
            category="OpenAI API/Memory",
            description="Returns a copy of the value stored under a key by a Save Memory node, usually during the previous queued run. Returns the Default input, or None (as if nothing was connected) when the key is empty or nothing (or None) is stored under it.",
            inputs=[
                io.String.Input(
                    id="key",
                    display_name="Key",
                    tooltip="The name under which the value was stored. Leading and trailing spaces are ignored.",
                    default="memory",
                ),
                io.Boolean.Input(
                    id="reset",
                    display_name="Reset",
                    tooltip="Wipe the value stored under this key and return nothing for this run. Untick it afterwards, otherwise the key is reset on every run.",
                    default=False,
                ),
                io.AnyType.Input(
                    id="default",
                    display_name="Default",
                    optional=True,
                    tooltip="Optional value returned instead of None when the key is empty, missing, reset or holds None, e.g. on the first run of a loop when the output feeds a required input",
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    id="value",
                    display_name="Value",
                    tooltip="A copy of the stored value, or the default input (None if not connected) if there is none",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> Any:
        # ComfyUI evaluates the fingerprints of the whole prompt before
        # executing any node: the version is the one left by the previous run.
        # A linked input (None here) is not known yet: always re-run then.
        reset = kwargs.get("reset")
        key = kwargs.get("key")
        if reset or reset is None or key is None:
            return float("nan")
        # Unchanged version: the cached output (and everything downstream)
        # is reused
        return MEMORY.version(normalize_key(key))

    @classmethod
    def execute(cls, key: str, reset: bool, default: Any = None) -> io.NodeOutput:
        # default is None when not connected: same as returning nothing
        normalized_key = normalize_key(key)
        if normalized_key == "":
            return io.NodeOutput(default)
        if reset:
            MEMORY.wipe(normalized_key)
            return io.NodeOutput(default)
        # A stored None counts as nothing stored, so default applies to it too
        _, value = MEMORY.load(normalized_key)
        return io.NodeOutput(default if value is None else value)
