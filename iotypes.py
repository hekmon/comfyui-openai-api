import json
from typing import Any

from comfy_api.latest import io
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


def check_history(history: Any, node: str) -> None:
    """A History input linked to a Load Memory node can receive any value: the
    key may hold something else than a conversation history."""
    if history is not None and not isinstance(history, HistoryPayload):
        raise ValueError(
            f"{node}: the History input received a {type(history).__name__} instead of a conversation "
            "history. If it comes from a Load Memory node, its key may hold another value "
            "(saved by another Save Memory node): use a dedicated key or tick Reset once.")


class OptionsPayload:
    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options = options

    def get_options_copy(self) -> dict[str, Any]:
        return self.options.copy() if self.options else {}

    def __str__(self) -> str:
        return json.dumps(self.options, indent=4)
