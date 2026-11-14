from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from comfy_api.latest import io

from .iotypes import ParamHistory, HistoryPayload


_INSTRUCTIONS_ROLES = ("system", "developer")


def trim_messages(messages: list[ChatCompletionMessageParam], max_turns: int) -> list[ChatCompletionMessageParam]:
    """Keep the leading system/developer message (if any) and the last
    max_turns turns. A turn starts at a user message and includes every
    message following it until the next user message, so an exchange is never
    split and the first kept non instructions message is always a user one.
    Returns a new list, the input one is not modified."""
    head: list[ChatCompletionMessageParam] = []
    rest = messages
    if rest and rest[0].get("role") in _INSTRUCTIONS_ROLES:
        head = [rest[0]]
        rest = rest[1:]
    turn_starts = [i for i, msg in enumerate(rest) if msg.get("role") == "user"]
    if not turn_starts:
        # No complete turn: drop the orphan replies
        return head
    start = turn_starts[-max_turns] if len(turn_starts) > max_turns else turn_starts[0]
    return head + list(rest[start:])


class TrimHistory(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_TrimHistory",
            display_name="OpenAI API - Trim History",
            category="OpenAI API",
            description="Limits a conversation history to its last turns, e.g. before storing it in memory in a looping workflow. The leading system/developer message is kept. A turn starts at a user message and includes the answers that follow it, so an exchange is never split.",
            inputs=[
                ParamHistory.Input(
                    id="history",
                    display_name="History",
                    tooltip="The conversation history to trim. Nothing (None) is passed through as is.",
                ),
                io.Int.Input(
                    id="max_turns",
                    display_name="Max Turns",
                    tooltip="Number of most recent turns (user message + answers) to keep",
                    default=10,
                    min=1,
                    max=10000,
                    display_mode=io.NumberDisplay.number,
                ),
            ],
            outputs=[
                ParamHistory.Output(
                    id="trimmed_history",
                    display_name="History",
                    tooltip="The trimmed conversation history",
                ),
            ],
        )

    @classmethod
    def execute(cls, history: HistoryPayload | None, max_turns: int) -> io.NodeOutput:
        # A Load Memory node returns None on the first run of a loop: keep it
        # as "no history" instead of failing
        if history is None:
            return io.NodeOutput(None)
        return io.NodeOutput(HistoryPayload(trim_messages(history.get_msgs_copy(), max_turns)))
