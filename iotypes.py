import json
from typing import Any

from comfy_api.latest import io
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


ParamClient = io.Custom("OAIAPI_CLIENT")
ParamHistory = io.Custom("OAIAPI_HISTORY")
ParamOptions = io.Custom("OAIAPI_OPTIONS")

# Roles of the leading instructions message of a conversation
INSTRUCTIONS_ROLES = ("system", "developer")


class HistoryPayload:
    def __init__(self, history: list[ChatCompletionMessageParam] | None = None) -> None:
        self.history = history

    def get_msgs_copy(self) -> list[ChatCompletionMessageParam]:
        return self.history.copy() if self.history else []

    def __str__(self) -> str:
        return json.dumps(self.history, indent=4)


def check_input(value: Any, expected: type, input_name: str, what: str, node: str, optional: bool = True) -> None:
    """An input linked to a Load Memory node (an AnyType output) can receive
    any value: the key may hold something else than what the input expects.
    Fails with a clear message instead of an error deep inside the node."""
    if (value is None and optional) or isinstance(value, expected):
        return
    received = type(value).__name__
    article = "an" if received[:1].lower() in "aeiou" else "a"
    raise ValueError(
        f"{node}: the {input_name} input received {article} {received} instead of {what}. "
        "If it comes from a Load Memory node, its key may hold another value "
        "(saved by another Save Memory node): use a dedicated key or tick Reset once.")


def check_history(history: Any, node: str) -> None:
    check_input(history, HistoryPayload, "History", "a conversation history", node)


def check_options(options: Any, node: str) -> None:
    check_input(options, OptionsPayload, "Options", "options (from the option nodes)", node)


def check_client(client: Any, node: str) -> None:
    check_input(client, AsyncOpenAI, "API Client", "an API client (from the Client node)", node, optional=False)


class OptionsPayload:
    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options = options

    def get_options_copy(self) -> dict[str, Any]:
        return self.options.copy() if self.options else {}

    def __str__(self) -> str:
        return json.dumps(self.options, indent=4)
