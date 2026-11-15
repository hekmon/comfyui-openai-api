from typing import Any

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from comfy_api.latest import io

from .iotypes import INSTRUCTIONS_ROLES, ParamHistory, HistoryPayload, check_history, check_input


def trim_messages(messages: list[ChatCompletionMessageParam], max_turns: int) -> list[ChatCompletionMessageParam]:
    """Keep the leading system/developer message (if any) and the last
    max_turns turns. A turn starts at a user message and includes every
    message following it until the next user message, so an exchange is never
    split and the first kept non instructions message is always a user one.
    Returns a new list, the input one is not modified."""
    head: list[ChatCompletionMessageParam] = []
    rest = messages
    if rest and rest[0].get("role") in INSTRUCTIONS_ROLES:
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
        check_history(history, "Trim History")
        return io.NodeOutput(HistoryPayload(trim_messages(history.get_msgs_copy(), max_turns)))


# Text of the part that replaces the content of a message left empty by
# Trim Images: an empty message could be rejected by the server and dropping
# it would break the user/assistant alternation
IMAGE_REMOVED_TEXT = "[image removed]"


def _is_image_part(part: Any) -> bool:
    # Chat Completion sends its images as image_url parts (PNG data URLs)
    return isinstance(part, dict) and part.get("type") == "image_url"


def count_images(messages: list[ChatCompletionMessageParam]) -> int:
    return sum(
        1
        for msg in messages
        if isinstance(msg.get("content"), list)
        for part in msg["content"]
        if _is_image_part(part)
    )


def trim_images(messages: list[ChatCompletionMessageParam], max_images: int,
                trim_to: int) -> list[ChatCompletionMessageParam]:
    """If the messages hold more than max_images images, remove the oldest
    ones so that trim_to remain. Only the image parts are removed: every
    message and every other part is kept, in order. A message left without
    any part gets an IMAGE_REMOVED_TEXT text part.
    Returns a new list. The input messages are not modified: a changed
    message is a new dict with a new content list, the others are shared."""
    count = count_images(messages)
    if count <= max_images:
        return list(messages)
    to_remove = count - trim_to
    trimmed: list[ChatCompletionMessageParam] = []
    for msg in messages:
        content = msg.get("content")
        if to_remove <= 0 or not isinstance(content, list):
            trimmed.append(msg)
            continue
        kept = []
        for part in content:
            if to_remove > 0 and _is_image_part(part):
                to_remove -= 1
            else:
                kept.append(part)
        if len(kept) == len(content):
            trimmed.append(msg)
            continue
        # Kept as a list of parts, like the multi modal messages of Chat
        # Completion: valid for every role
        trimmed.append({**msg, "content": kept or [{"type": "text", "text": IMAGE_REMOVED_TEXT}]})
    return trimmed


def check_trim_images_inputs(max_images: int, trim_to: int) -> str | None:
    if max_images < 0 or trim_to < 0:
        return f"Max Images ({max_images}) and Trim To ({trim_to}) must not be negative"
    if trim_to > max_images:
        return (f"Trim To ({trim_to}) must not be greater than Max Images ({max_images}): "
                "it is the number of images kept once the history holds more than Max Images")
    return None


class TrimImages(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_TrimImages",
            display_name="OpenAI API - Trim Images",
            category="OpenAI API",
            description="Limits the number of images of a conversation history, for servers that accept only a few images per conversation, e.g. in a looping workflow: Load Memory -> Trim Images -> Chat Completion (with a new image) -> Save Memory. Only the images already in the history are counted, not the ones the Chat Completion node adds in the current turn. Once they are more than Max Images, the oldest are removed so that Trim To remain. Only the images are removed: every message and its text is kept, a message left empty gets the text [image removed]. Removing an image breaks the server prompt cache: see Trim To.",
            inputs=[
                ParamHistory.Input(
                    id="history",
                    display_name="History",
                    tooltip="The conversation history to trim. Nothing (None) is passed through as is.",
                ),
                io.Int.Input(
                    id="max_images",
                    display_name="Max Images",
                    tooltip="Maximum number of images in the history: up to this number, the history is passed through unchanged. Only the images already in the history are counted, not the ones the Chat Completion node adds in the current turn: for a server limited to 8 images per conversation and 1 new image per turn, set 7.",
                    default=7,
                    min=0,
                    max=10000,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    id="trim_to",
                    display_name="Trim To",
                    tooltip="Number of most recent images kept once the history holds more than Max Images: the oldest are removed. Must not be greater than Max Images. Removing an old image changes the start of the conversation: the server can not reuse its prompt (prefix) cache and processes the whole conversation again, re-encoding the remaining images. Equal to Max Images, this happens on every turn once the limit is reached. Lower, it happens less often, at the cost of fewer images kept: e.g. Max Images 7 and Trim To 3 with 1 new image per turn trims once every 5 turns.",
                    default=7,
                    min=0,
                    max=10000,
                    display_mode=io.NumberDisplay.number,
                ),
            ],
            outputs=[
                ParamHistory.Output(
                    id="trimmed_history",
                    display_name="History",
                    tooltip="The history without its oldest images, or unchanged",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, max_images: int | None, trim_to: int | None) -> bool | str:
        # None when coming from another node: checked at execution
        if max_images is None or trim_to is None:
            return True
        return check_trim_images_inputs(max_images, trim_to) or True

    @classmethod
    def execute(cls, history: HistoryPayload | None, max_images: int, trim_to: int) -> io.NodeOutput:
        check_input(max_images, int, "Max Images", "a whole number", "Trim Images", optional=False)
        check_input(trim_to, int, "Trim To", "a whole number", "Trim Images", optional=False)
        error = check_trim_images_inputs(max_images, trim_to)
        if error is not None:
            raise ValueError(f"Trim Images: {error}")
        # None on the first run of a loop (Load Memory): no history
        if history is None:
            return io.NodeOutput(None)
        check_history(history, "Trim Images")
        return io.NodeOutput(HistoryPayload(trim_images(history.get_msgs_copy(), max_images, trim_to)))
