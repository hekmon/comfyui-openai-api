import json
from typing import Any

from comfy_api.latest import io
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


ParamClient = io.Custom("OAIAPI_CLIENT")
ParamHistory = io.Custom("OAIAPI_HISTORY")
ParamOptions = io.Custom("OAIAPI_OPTIONS")


class HistoryPayload:
    def __init__(self, history: list[ChatCompletionMessageParam] | None = None) -> None:
        self.history = history

    def get_msgs_copy(self) -> list[ChatCompletionMessageParam]:
        return self.history.copy() if self.history else []

    def __str__(self) -> str:
        return json.dumps(self.history, indent=4)


class OptionsPayload:
    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options = options

    def get_options_copy(self) -> dict[str, Any]:
        return self.options.copy() if self.options else {}

    def __str__(self) -> str:
        return json.dumps(self.options, indent=4)
