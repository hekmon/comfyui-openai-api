import asyncio
import base64
from io import BytesIO
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np
from PIL import Image
import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion as ChatCompletionResponse
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam

from comfy_api.latest import io, ui

from . import aioloop
from .apierrors import describe_api_error, scrub_exception
from .interrupt import run_interruptible
from .iotypes import (INSTRUCTIONS_ROLES, ParamClient, ParamHistory, ParamOptions, HistoryPayload, OptionsPayload,
                      check_client, check_history, check_input, check_options)


logger = logging.getLogger(__name__)


def comfy_image_to_base64_png_url(image: torch.Tensor) -> str:
    # Taken from the SaveImage ComfyUI node, convert the tensor into a regular image
    i = np.multiply(255., image.cpu().numpy())
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    # Encode the image as PNG in base64 format
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    b64_png = base64.b64encode(buffer.getvalue())
    # Return the formated string URL
    return f"data:image/png;base64,{b64_png.decode('utf-8')}"


def _encode_images(images: torch.Tensor, stop: threading.Event) -> list[str]:
    urls: list[str] = []
    for image in images:  # iterating a [B,H,W,C] tensor yields [H,W,C] images
        if stop.is_set():
            break  # the caller gave up (Cancel, prompt failure): result unused
        urls.append(comfy_image_to_base64_png_url(image))
    return urls


async def encode_images(images: torch.Tensor) -> list[str]:
    """The PNG data URLs of a batch of images, encoded in a worker thread: the
    encoding of a big batch takes seconds, and on the prompt event loop it
    would stall every other node running meanwhile (and the Cancel checks).
    The images are only read.

    Stops at Cancel like a request (see run_interruptible). A running thread
    can not be cancelled: the stop event ends it after the current image, so
    that a Cancel does not wait for the whole batch to end the prompt."""
    stop = threading.Event()
    try:
        return await run_interruptible(asyncio.to_thread(_encode_images, images, stop))
    finally:
        stop.set()


def extract_response_content(completion: ChatCompletionResponse) -> str:
    # Some backends or proxies answer with an HTTP 200 but no usable choice
    choices = getattr(completion, "choices", None)
    if not choices:
        raise ValueError(
            "the API returned no choices in its response (empty or missing "
            "'choices' field): check the model name and the server logs")
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        raise ValueError("the API returned a choice without a message")
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ValueError(f"the model refused to answer: {refusal}")
    finish_reason = getattr(choice, "finish_reason", None)
    content = getattr(message, "content", None)
    if content is None:
        # e.g. a reasoning model that spent its whole token budget thinking
        # (finish_reason "length") or a tool call answer
        logger.warning(
            "the API returned no text content (finish_reason: %s), "
            "using an empty response", finish_reason)
        return ""
    return content


def _format_token_details(details: Any, fields: list[tuple[str, str]]) -> str:
    # Only list the strictly positive counters, and only add the parentheses
    # if there is at least one of them
    parts = []
    for attr, label in fields:
        value = getattr(details, attr, None)
        if isinstance(value, int) and value > 0:
            parts.append(f"{label}: {value}")
    return f" ({', '.join(parts)})" if parts else ""


def _format_token_count(value: Any) -> str:
    return str(value) if value is not None else "n/a"


def format_usage(usage: CompletionUsage | None) -> str | None:
    if usage is None:
        return None
    # Some OpenAI-compatible backends return a usage object without some of
    # these fields (even the token totals): always use getattr with a default
    text = "Prompt tokens: " + _format_token_count(getattr(usage, "prompt_tokens", None))
    text += _format_token_details(
        getattr(usage, "prompt_tokens_details", None),
        [
            ("audio_tokens", "audio"),
            ("cached_tokens", "cached"),
        ],
    )
    text += "\nCompletions tokens: " + _format_token_count(getattr(usage, "completion_tokens", None))
    text += _format_token_details(
        getattr(usage, "completion_tokens_details", None),
        [
            ("audio_tokens", "audio"),
            ("reasoning_tokens", "reasoning"),
            ("accepted_prediction_tokens", "prediction accepted"),
            ("rejected_prediction_tokens", "prediction rejected"),
        ],
    )
    return text


# Request fields set by the option nodes: sent as SDK arguments, and omitted
# when unset (or null in an Extra Body) so providers are not sent null values
_REQUEST_OPTIONS = (
    "seed",  # deprecated by OpenAI, still honoured by most local backends
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    # A string: "none" is a real value that must be sent, only a missing
    # Reasoning Effort node omits the field
    "reasoning_effort",
)
# Set by the Developer Role node, read here and never sent
_DEVELOPER_ROLE_KEY = "use_developer_role"


@dataclass
class _RequestOptions:
    fields: dict[str, Any]  # request fields with a value, in _REQUEST_OPTIONS order
    extra_body: dict[str, Any]  # every other key (Extra Body node...), sent as is
    developer_role_set: bool  # a Developer Role node (or key) is present
    use_developer_role: Any  # its value, used as a boolean


def _split_options(options: OptionsPayload | None) -> _RequestOptions:
    extra_body = options.get_options_copy() if options is not None else {}
    fields: dict[str, Any] = {}
    for key in _REQUEST_OPTIONS:
        value = extra_body.pop(key, None)
        if value is not None:
            fields[key] = value
    # Its presence counts, even with a null value (from an Extra Body)
    developer_role_set = _DEVELOPER_ROLE_KEY in extra_body
    use_developer_role = extra_body.pop(_DEVELOPER_ROLE_KEY, False)
    return _RequestOptions(fields, extra_body, developer_role_set, use_developer_role)


def _instructions_message(content: str, developer: bool) -> ChatCompletionMessageParam:
    # Two literals: the message types need a literal role
    if developer:
        return {"role": "developer", "content": content}
    return {"role": "system", "content": content}


def _build_messages(history: HistoryPayload | None, system_prompt: str | None,
                    options: _RequestOptions) -> list[ChatCompletionMessageParam]:
    """The messages that come before the new user message: a copy of the
    history, with the system prompt set as its instructions message (replacing
    the existing one, if any) or the existing one converted to the role of the
    Developer Role option."""
    messages = history.get_msgs_copy() if history is not None else []
    has_instructions = bool(messages) and messages[0].get("role") in INSTRUCTIONS_ROLES
    if system_prompt:
        instructions = _instructions_message(system_prompt, bool(options.use_developer_role))
        if has_instructions:
            messages[0] = instructions
        else:
            messages.insert(0, instructions)
    elif has_instructions and options.developer_role_set:
        # No new instructions but the Developer Role option is explicitly
        # set: convert the existing instructions message role to match it.
        # Build a new dict, the history messages are shared with the
        # (cached) upstream node output and must not be mutated.
        wanted_role = "developer" if options.use_developer_role else "system"
        if messages[0].get("role") != wanted_role:
            messages[0] = {**messages[0], "role": wanted_role}
    return messages


def _user_message(prompt: str, image_urls: list[str] | None) -> ChatCompletionMessageParam:
    if image_urls is None:
        return {"role": "user", "content": prompt}
    # Multi modal content: every image of the batch, then the prompt once (it
    # must not be repeated for each image)
    content: list[ChatCompletionContentPartParam] = [
        {"type": "image_url", "image_url": {"url": url}} for url in image_urls
    ]
    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


class ChatCompletion(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_ChatCompletion",
            display_name="OpenAI API - Chat Completion",
            category="OpenAI API",
            description="Generates text responses using OpenAI's chat completion API. Be sure to indicate a Vision Language Model if you are using an image input.",
            inputs=[
                ParamClient.Input(
                    id="client",
                    display_name="API Client",
                    tooltip="The OpenAI API client to use to perform the request"
                ),
                io.String.Input(
                    id="model",
                    display_name="Model",
                    tooltip="The model to use for generating text",
                    placeholder="Model name",
                ),
                io.Boolean.Input(
                    id="force_regen",
                    display_name="Force Regen",
                    tooltip="Set to true to always request a new text generation even if no widget input values have changed (no cache). If this input is linked to another node, the generation always runs again as its value is not known when ComfyUI checks the cache.",
                    default=False,
                ),
                io.String.Input(
                    id="prompt",
                    display_name="Prompt",
                    tooltip="The prompt to use for generating text",
                    multiline=True,
                    placeholder="user prompt is mandatory",
                ),
                io.String.Input(
                    id="system_prompt",
                    display_name="System Prompt",
                    optional=True,
                    tooltip="The system prompt to send along with the user prompt",
                    multiline=True,
                    placeholder="system/developer prompt is optional",
                ),
                ParamHistory.Input(
                    id="history",
                    display_name="History",
                    optional=True,
                    tooltip="Previous conversation history",
                ),
                ParamOptions.Input(
                    id="options",
                    display_name="Options",
                    optional=True,
                    tooltip="Additional options to pass with the request",
                ),
                io.Image.Input(
                    id="images",
                    display_name="image(s)",
                    optional=True,
                    tooltip="Image(s) to include in the request",
                ),
            ],
            outputs=[
                io.String.Output(
                    id="response",
                    display_name="Response",
                    tooltip="Generated text response",
                ),
                ParamHistory.Output(
                    id="complete_chatcompletion",
                    display_name="History",
                    tooltip="Conversation history",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, model: str, prompt: str) -> bool | str:
        if model == "":
            return "model must be specified"
        if prompt == "":
            return "prompt must be specified"
        return True

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> str:
        # ComfyUI already includes every widget value and upstream link in the
        # cache key, this fingerprint only has to break the cache when needed.
        # A linked force_regen is received as None here: its value is not known
        # yet at fingerprinting time, so err on the side of regenerating.
        force_regen = kwargs.get("force_regen", False)
        if force_regen is None or force_regen:
            return str(time.time())  # Use timestamp for always refresh
        return ""  # constant: let the regular ComfyUI cache apply

    @classmethod
    async def execute(cls,
                      client: AsyncOpenAI,
                      model: str,
                      prompt: str,
                      system_prompt: str | None = None,
                      history: HistoryPayload | None = None,
                      options: OptionsPayload | None = None,
                      images: torch.Tensor | None = None,
                      force_regen: bool = False,
                ) -> io.NodeOutput:
        # Linked inputs are not checked by validate_inputs (None there)
        if not isinstance(prompt, str) or prompt == "":
            raise ValueError(
                "Chat Completion: the prompt is empty (or not a text). If it comes from a "
                "Load Memory node, connect its Default input for the first run.")
        check_input(system_prompt, str, "System Prompt", "a text", "Chat Completion")
        check_history(history, "Chat Completion")
        check_options(options, "Chat Completion")
        check_client(client, "Chat Completion")
        request_options = _split_options(options)
        # Never send both token limits at once. Checked before encoding the
        # images, which can take a while.
        if "max_tokens" in request_options.fields and "max_completion_tokens" in request_options.fields:
            raise ValueError(
                "max_tokens and max_completion_tokens cannot both be set. "
                "Use only one: max_tokens for older models and providers, "
                "or max_completion_tokens for newer OpenAI chat models."
            )
        image_urls = await encode_images(images) if images is not None else None
        messages = _build_messages(history, system_prompt, request_options)
        messages.append(_user_message(prompt, image_urls))
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "n": 1,
            **request_options.fields,
        }
        if request_options.extra_body:
            request_kwargs["extra_body"] = request_options.extra_body
        try:
            # Sent from the API client loop, where the client lives: its
            # connections are shared with the other nodes using this client
            completion = await run_interruptible(
                aioloop.submit(client.chat.completions.create(**request_kwargs)))
        except (openai.APIStatusError, openai.APIConnectionError) as err:
            # The chained original exception is printed in the ComfyUI log:
            # strip any echoed credentials from it too
            scrub_exception(err, client)
            raise RuntimeError(f"Chat completion request failed: {describe_api_error(err, client)}") from err
        content = extract_response_content(completion)
        # Add the response to the history (content is never None here, a None
        # assistant content would be rejected by the server on the next turn)
        messages.append(
            {
                "role": "assistant",
                "content": content
            }
        )
        # Handle usage stats as text preview
        stats = format_usage(completion.usage)
        # Also printed to the console, after the log line of the HTTP request
        if stats:
            print(stats)
        # Return the response and the history and the stats for the UI
        return io.NodeOutput(
            content,
            HistoryPayload(messages),
            ui=ui.PreviewText(stats or ""),
        )
