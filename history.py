import logging
from typing import Any

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from comfy_api.latest import io

from .iotypes import INSTRUCTIONS_ROLES, ParamHistory, HistoryPayload, check_history, check_input


logger = logging.getLogger(__name__)

# Bounds of the Trim History widgets (-1: see the tooltips)
_MAX_VALUE = 10000

# Input values already warned about (capped or ignored Keep values): one
# warning per combination, not one per run of a looping workflow
_warned_inputs: set[tuple[int, int, int, int]] = set()


def _split_turns(messages: list[ChatCompletionMessageParam]
                 ) -> tuple[list[ChatCompletionMessageParam], list[ChatCompletionMessageParam], list[int]]:
    """The leading system/developer message (if any, as a list), the other
    messages and the index in them of every user message (turn start)."""
    head: list[ChatCompletionMessageParam] = []
    rest = messages
    if rest and rest[0].get("role") in INSTRUCTIONS_ROLES:
        head = [rest[0]]
        rest = rest[1:]
    return head, rest, [i for i, msg in enumerate(rest) if msg.get("role") == "user"]


def trim_messages(messages: list[ChatCompletionMessageParam], max_turns: int) -> list[ChatCompletionMessageParam]:
    """Keep the leading system/developer message (if any) and the last
    max_turns turns (0: none). A turn starts at a user message and includes
    every message following it until the next user message, so an exchange is
    never split and the first kept non instructions message is always a user
    one. Returns a new list, the input one is not modified."""
    head, rest, turn_starts = _split_turns(messages)
    if not turn_starts or max_turns <= 0:
        # No complete turn (drop the orphan replies) or no turn to keep
        return head
    start = turn_starts[-max_turns] if len(turn_starts) > max_turns else turn_starts[0]
    return head + list(rest[start:])


# Text of the part that replaces the content of a message left empty by
# the removal of its images: an empty message could be rejected by the
# server and dropping it would break the user/assistant alternation
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


def keep_last_images(messages: list[ChatCompletionMessageParam], keep: int) -> list[ChatCompletionMessageParam]:
    """Remove the oldest images so that at most keep remain. Only the image
    parts are removed: every message and every other part is kept, in order.
    A message left without any part gets an IMAGE_REMOVED_TEXT text part.
    Returns a new list. The input messages are not modified: a changed
    message is a new dict with a new content list, the others are shared."""
    to_remove = count_images(messages) - keep
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


def effective_keep(max_turns: int, max_images: int, keep_turns: int, keep_images: int) -> tuple[int, int, list[str]]:
    """The Keep values actually used by a cut, and why they differ from the
    inputs. -1 means the Max value. A Keep value above its Max value is capped
    at it, and Keep Images is ignored without an image limit (Max Images -1):
    the Keep inputs are advanced (often hidden) inputs and the node runs in
    unattended loops, so these are not errors."""
    notes = []
    turns = max_turns if keep_turns == -1 else keep_turns
    if turns > max_turns:
        notes.append(f"Keep Turns ({keep_turns}) is greater than Max Turns ({max_turns}): capped at {max_turns}")
        turns = max_turns
    images = max_images if keep_images == -1 else keep_images
    if max_images == -1:
        if keep_images != -1:
            notes.append(f"Keep Images ({keep_images}) is ignored: Max Images is -1 (no image limit)")
        images = -1
    elif images > max_images:
        notes.append(f"Keep Images ({keep_images}) is greater than Max Images ({max_images}): capped at {max_images}")
        images = max_images
    return turns, images, notes


def trim_history(messages: list[ChatCompletionMessageParam], max_turns: int, max_images: int = -1,
                 keep_turns: int = -1, keep_images: int = -1) -> list[ChatCompletionMessageParam]:
    """The coordinated cut of Trim History. Nothing is cut while the history
    holds at most max_turns turns and (unless max_images is -1) at most
    max_images images. Above either limit, the oldest turns are dropped so
    that keep_turns remain, then the oldest images of the remaining turns so
    that keep_images remain (never when max_images is -1). The Keep values
    are the effective ones (effective_keep). Either way, the leading
    system/developer message is kept and the orphan replies before the first
    user message are dropped (trim_messages).
    Returns a new list, the input messages are not modified."""
    keep_turns, keep_images, _ = effective_keep(max_turns, max_images, keep_turns, keep_images)
    _, rest, turn_starts = _split_turns(messages)
    # Images of the turns (the orphan replies before the first user message
    # are dropped either way, their images do not count)
    images = count_images(rest[turn_starts[0]:]) if turn_starts else 0
    cut = len(turn_starts) > max_turns or 0 <= max_images < images
    trimmed = trim_messages(messages, keep_turns if cut else max_turns)
    if cut and max_images >= 0:
        trimmed = keep_last_images(trimmed, keep_images)
    return trimmed


def check_trim_inputs(max_turns: int, max_images: int, keep_turns: int, keep_images: int) -> str | None:
    """Only the impossible values: the Keep values above the Max ones are
    capped (effective_keep), not rejected."""
    if max_turns < 1:
        return f"Max Turns ({max_turns}) must be at least 1"
    for name, value in (("Max Images", max_images), ("Keep Turns", keep_turns), ("Keep Images", keep_images)):
        if value < -1:
            return f"{name} ({value}) must be -1 or at least 0"
    return None


DESCRIPTION = (
    "Limits a conversation history to its last turns and, optionally, its last images. Why a single node: "
    "every cut changes the start of the conversation, so the server loses its prompt cache and reprocesses "
    "everything, images included; cutting turns and images together costs one cache loss per cut instead of "
    "two. Nothing changes up to Max Turns and Max Images; above either, the oldest turns are dropped down to "
    "Keep Turns, then the oldest images down to Keep Images. The Keep values (advanced inputs) leave a margin "
    "so cuts happen less often (e.g. Max Images 7, Keep Images 3, 1 image per run: one cut every 5 runs). "
    "Place it right after Load Memory (Load Memory -> Trim History -> Chat Completion -> Save Memory) and "
    "size it so that what it keeps plus what one run adds fits the server. The leading system message is "
    "kept, an exchange is never split, and a message emptied of its images gets the text [image removed]."
)
MAX_TURNS_TOOLTIP = (
    "Maximum number of turns (a user message and its answers) in the history: up to this number, nothing is "
    "cut. Above it, the oldest turns are dropped so that Keep Turns remain. Only the turns already in the "
    "history are counted, not the ones the next Chat Completion nodes add in this run: to send at most 10 turns "
    "with 1 new turn per run, set 9."
)
MAX_IMAGES_TOOLTIP = (
    "Maximum number of images in the history. -1: no image limit. 0: keep no image (e.g. only the text of the "
    "image analyses). Up to this number, nothing is cut; above it, a cut keeps the Keep Images most recent "
    "images (the text stays). Only the images already in the history are counted, not the ones the next Chat "
    "Completion adds in this run: for a server limited to 8 images and 1 new image per run, set 7."
)
KEEP_TURNS_TOOLTIP = (
    "Number of turns kept by a cut. -1: same as Max Turns. A cut happens when the history holds more than Max "
    "Turns turns or more than Max Images images. Lower than Max Turns, it leaves a margin: the history grows "
    "again for several runs before the next cut, so the server loses its prompt cache less often, at the cost "
    "of a shorter history right after a cut. 0 keeps no turn. Capped at Max Turns."
)
KEEP_IMAGES_TOOLTIP = (
    "Number of images kept by a cut, the most recent ones. -1: same as Max Images. Lower than Max Images, it "
    "leaves a margin so that cuts happen less often: e.g. Max Images 7 and Keep Images 3 with 1 new image per "
    "run cut once every 5 runs instead of every run. Capped at Max Images, ignored when Max Images is -1."
)


class TrimHistory(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_TrimHistory",
            display_name="OpenAI API - Trim History",
            category="OpenAI API",
            description=DESCRIPTION,
            inputs=[
                ParamHistory.Input(
                    id="history",
                    display_name="History",
                    tooltip="The conversation history to trim. Nothing (None) is passed through as is.",
                ),
                io.Int.Input(
                    id="max_turns",
                    display_name="Max Turns",
                    tooltip=MAX_TURNS_TOOLTIP,
                    default=10,
                    min=1,
                    max=_MAX_VALUE,
                    display_mode=io.NumberDisplay.number,
                ),
                # Optional: the workflows saved with Trim History 3.1.0 (API
                # format) only hold max_turns. Appended after it: the UI
                # format stores the widget values by position.
                io.Int.Input(
                    id="max_images",
                    display_name="Max Images",
                    optional=True,
                    tooltip=MAX_IMAGES_TOOLTIP,
                    default=-1,
                    min=-1,
                    max=_MAX_VALUE,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    id="keep_turns",
                    display_name="Keep Turns",
                    optional=True,
                    advanced=True,
                    tooltip=KEEP_TURNS_TOOLTIP,
                    default=-1,
                    min=-1,
                    max=_MAX_VALUE,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    id="keep_images",
                    display_name="Keep Images",
                    optional=True,
                    advanced=True,
                    tooltip=KEEP_IMAGES_TOOLTIP,
                    default=-1,
                    min=-1,
                    max=_MAX_VALUE,
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
    def validate_inputs(cls, max_turns: int | None, max_images: int | None = -1,
                        keep_turns: int | None = -1, keep_images: int | None = -1) -> bool | str:
        values = (max_turns, max_images, keep_turns, keep_images)
        # None when coming from another node: checked at execution
        if None in values:
            return True
        # Listing an input here disables the ComfyUI min/max check of its
        # widget value: done here instead
        if max(values) > _MAX_VALUE:
            return f"Max Turns, Max Images, Keep Turns and Keep Images must not be greater than {_MAX_VALUE}"
        return check_trim_inputs(*values) or True

    @classmethod
    def execute(cls, history: HistoryPayload | None, max_turns: int, max_images: int = -1,
                keep_turns: int = -1, keep_images: int = -1) -> io.NodeOutput:
        for value, name in ((max_turns, "Max Turns"), (max_images, "Max Images"),
                            (keep_turns, "Keep Turns"), (keep_images, "Keep Images")):
            check_input(value, int, name, "a whole number", "Trim History", optional=False)
        error = check_trim_inputs(max_turns, max_images, keep_turns, keep_images)
        if error is not None:
            raise ValueError(f"Trim History: {error}")
        _, _, notes = effective_keep(max_turns, max_images, keep_turns, keep_images)
        values = (max_turns, max_images, keep_turns, keep_images)
        if notes and values not in _warned_inputs:
            _warned_inputs.add(values)
            logger.warning("Trim History: %s", "; ".join(notes))
        # A Load Memory node returns None on the first run of a loop: keep it
        # as "no history" instead of failing
        if history is None:
            return io.NodeOutput(None)
        check_history(history, "Trim History")
        return io.NodeOutput(HistoryPayload(trim_history(history.get_msgs_copy(), max_turns, max_images,
                                                         keep_turns, keep_images)))
