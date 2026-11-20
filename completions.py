import asyncio
import base64
from io import BytesIO
import threading
import time
from dataclasses import dataclass
from typing import Any, cast

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
from .logs import get_logger
from .apierrors import describe_api_error, scrub_exception
from .interrupt import run_interruptible
from .iotypes import (DEVELOPER_ROLE_KEY, INSTRUCTIONS_ROLES, ParamClient, ParamHistory, ParamOptions, HistoryPayload, OptionsPayload,
                      check_client, check_history, check_input, check_options)


logger = get_logger(__name__)


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


# Extra assistant message fields holding the reasoning of reasoning models,
# in fallback order: "reasoning_content" (llama.cpp, SGLang, LM Studio,
# tabbyAPI, DeepSeek...) then "reasoning" (recent vLLM, Ollama, OpenRouter,
# Groq...). Older vLLM versions send both.
REASONING_KEYS = ("reasoning_content", "reasoning")
# The structured reasoning of the relays (OpenRouter, Vercel AI Gateway) for
# the OpenAI, Anthropic and Gemini models, along with a "reasoning" text: a
# list of text, summary and encrypted blocks (opaque, tied to the provider
# and model that made them), to send back unmodified and in order. The
# OpenAI API itself returns no reasoning on chat completions.
REASONING_DETAILS_KEY = "reasoning_details"
# Every reasoning field kept in the history (see Send Reasoning)
HISTORY_REASONING_KEYS = (*REASONING_KEYS, REASONING_DETAILS_KEY)


def extract_reasoning(message: Any) -> dict[str, Any]:
    """The reasoning fields of a response message, under the key(s) received,
    in HISTORY_REASONING_KEYS order. Only the non empty texts (and non empty
    reasoning_details lists, kept as is) are kept: the history mirrors them,
    and a null or empty field sent back can be rejected (llama.cpp fails on a
    null reasoning_content)."""
    # Not declared by the SDK models: read from the extra fields they keep
    fields: dict[str, Any] = {}
    for key in REASONING_KEYS:
        value = getattr(message, key, None)
        if isinstance(value, str) and value:
            fields[key] = value
    details = getattr(message, REASONING_DETAILS_KEY, None)
    if isinstance(details, list) and details:
        fields[REASONING_DETAILS_KEY] = details
    return fields


def _detail_blocks(reasoning: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in reasoning.get(REASONING_DETAILS_KEY, []) if isinstance(b, dict)]


def reasoning_text(reasoning: dict[str, Any]) -> str | None:
    """The reasoning to show: the first received text field in REASONING_KEYS
    order (servers sending both send the same text twice), else the text and
    summary blocks of reasoning_details."""
    text = next((reasoning[key] for key in REASONING_KEYS if key in reasoning), None)
    if text is not None:
        return text
    parts = []
    for block in _detail_blocks(reasoning):
        value = block.get("text") if block.get("type") == "reasoning.text" else (
            block.get("summary") if block.get("type") == "reasoning.summary" else None)
        if isinstance(value, str) and value:
            parts.append(value)
    return "\n\n".join(parts) or None


def encrypted_reasoning(reasoning: dict[str, Any]) -> tuple[int, int]:
    """The number of encrypted reasoning blocks and their total size in
    characters: only these are shown, never their content."""
    sizes = [len(str(b.get("data") or "")) for b in _detail_blocks(reasoning)
             if b.get("type") == "reasoning.encrypted"]
    return len(sizes), sum(sizes)


@dataclass
class ResponseContent:
    content: str  # never None, empty if the API returned no text
    reasoning: dict[str, Any]  # see extract_reasoning, empty without reasoning


def extract_response_content(completion: ChatCompletionResponse) -> ResponseContent:
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
    reasoning = extract_reasoning(message)
    if content is None:
        if reasoning:
            # Most likely the token budget ran out before the end of the
            # reasoning (finish_reason "length")
            logger.warning(
                "Chat Completion: the API returned no text content (finish_reason: %s) but a "
                "reasoning: the model probably spent its token budget reasoning, "
                "raise the max tokens (max_tokens or max_completion_tokens). "
                "Using an empty response", finish_reason)
        else:
            # e.g. a tool call answer
            logger.warning(
                "Chat Completion: the API returned no text content (finish_reason: %s), "
                "using an empty response", finish_reason)
        content = ""
    return ResponseContent(content, reasoning)


def _assistant_message(response: ResponseContent) -> ChatCompletionMessageParam:
    """The response as a history message. The reasoning fields are kept under
    the key(s) received: sent back as is on the next turns, the server and its
    chat template decide what to do with them (a server may reject the key of
    another one, e.g. Groq and reasoning_content). The content is never None:
    a None assistant content would be rejected on the next turn."""
    # Extra keys, not declared by the SDK message types
    return cast(ChatCompletionMessageParam,
                {"role": "assistant", "content": response.content, **response.reasoning})


def _format_token_count(value: Any) -> str:
    return str(value) if value is not None else "n/a"


def _field(obj: Any, path: tuple[str, ...]) -> Any:
    # Undeclared fields are kept by the SDK models (as attributes) and nested
    # undeclared objects arrive as plain dicts
    for name in path:
        obj = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def _first_count(obj: Any, paths: tuple[tuple[str, ...], ...]) -> int | None:
    """The first token counter found in paths, None if none is reported.
    Servers sending several of them send the same number twice: never added."""
    for path in paths:
        value = _field(obj, path)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


# Prompt cache reads, in fallback order: the OpenAI field (OpenAI, llama.cpp,
# vLLM and SGLang with their cache report flag, Ollama, tabbyAPI, Unsloth
# Studio, OpenRouter...), DeepSeek, Anthropic style proxies (e.g. LiteLLM),
# older Kimi and Together
_CACHE_READ_FIELDS = (
    ("prompt_tokens_details", "cached_tokens"),
    ("prompt_cache_hit_tokens",),
    ("cache_read_input_tokens",),
    ("cached_tokens",),
)
# Prompt cache writes: OpenAI, OpenRouter and Kimi, vLLM (newly prefilled
# tokens stored in its prefix cache), Anthropic style proxies
_CACHE_WRITE_FIELDS = (
    ("prompt_tokens_details", "cache_write_tokens"),
    ("prompt_tokens_details", "created_cache_tokens"),
    ("cache_creation_input_tokens",),
)


def format_usage(usage: CompletionUsage | None, timings: Any = None) -> str | None:
    """The usage stats as text. timings: the llama.cpp style timings object
    of the response, if any, whose cache_n is the only cache counter of the
    llama.cpp builds before March 2026."""
    if usage is None:
        return None
    # Some OpenAI-compatible backends return a usage object without some of
    # these fields (even the token totals): always use getattr with a default
    text = "Prompt tokens: " + _format_token_count(getattr(usage, "prompt_tokens", None))
    cache_read = _first_count(usage, _CACHE_READ_FIELDS)
    if cache_read is None and isinstance(timings, dict):
        # Never timings.prompt_n: the uncached tokens for llama.cpp, the
        # whole prompt for Ollama
        cache_read = _first_count(timings, (("cache_n",),))
    cache_write = _first_count(usage, _CACHE_WRITE_FIELDS)
    prompt_parts = []
    # A reported 0 is shown: it tells a cache miss from a server that does
    # not report its cache (vLLM and SGLang need a flag for it)
    if cache_read is not None:
        prompt_parts.append(f"cache read: {cache_read}")
    # The rest of the prompt, processed by the server, only when it reports
    # its cache: the DeepSeek counter, else computed. The cache writes are
    # processed tokens too, part of it. Never negative: the counters of some
    # providers are rounded or overlap.
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    uncached = _first_count(usage, (("prompt_cache_miss_tokens",),))
    if (uncached is None and cache_read is not None and isinstance(prompt_tokens, int)
            and not isinstance(prompt_tokens, bool)):
        uncached = max(prompt_tokens - cache_read, 0)
    if uncached is not None:
        prompt_parts.append(f"uncached: {uncached}")
    # After uncached, which includes it: what the server stored of it
    if cache_write:
        prompt_parts.append(f"cache write: {cache_write}")
    audio = _first_count(usage, (("prompt_tokens_details", "audio_tokens"),))
    if audio:
        prompt_parts.append(f"audio: {audio}")
    if prompt_parts:
        text += f" ({', '.join(prompt_parts)})"
    completion_tokens = getattr(usage, "completion_tokens", None)
    text += "\nCompletions tokens: " + _format_token_count(completion_tokens)
    details = getattr(usage, "completion_tokens_details", None)
    completion_parts = []
    reasoning = _first_count(details, (("reasoning_tokens",),))
    audio = _first_count(details, (("audio_tokens",),))
    if reasoning:
        completion_parts.append(f"reasoning: {reasoning}")
        # The rest of the completion: the answer itself, so that nobody has
        # to subtract (audio tokens are part of the completion too)
        if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool):
            completion_parts.append(f"content: {max(completion_tokens - reasoning - (audio or 0), 0)}")
    if audio:
        completion_parts.append(f"audio: {audio}")
    for attr, label in (("accepted_prediction_tokens", "prediction accepted"),
                        ("rejected_prediction_tokens", "prediction rejected")):
        value = _first_count(details, ((attr,),))
        if value:
            completion_parts.append(f"{label}: {value}")
    if completion_parts:
        text += f" ({', '.join(completion_parts)})"
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


@dataclass
class _RequestOptions:
    fields: dict[str, Any]  # request fields with a value, in _REQUEST_OPTIONS order
    extra_body: dict[str, Any]  # every other key (Extra Body node...), sent as is
    # "system" or "developer" from a Developer Role node (or key), None without
    instructions_role: str | None


def _split_options(options: OptionsPayload | None) -> _RequestOptions:
    extra_body = options.get_options_copy() if options is not None else {}
    fields: dict[str, Any] = {}
    for key in _REQUEST_OPTIONS:
        value = extra_body.pop(key, None)
        if value is not None:
            fields[key] = value
    instructions_role = None
    # Its presence counts, even with a null value (from an Extra Body)
    if DEVELOPER_ROLE_KEY in extra_body:
        instructions_role = "developer" if extra_body.pop(DEVELOPER_ROLE_KEY) else "system"
    return _RequestOptions(fields, extra_body, instructions_role)


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
    role = options.instructions_role
    if system_prompt:
        instructions = _instructions_message(system_prompt, role == "developer")
        if has_instructions:
            messages[0] = instructions
        else:
            messages.insert(0, instructions)
    elif has_instructions and role is not None and messages[0].get("role") != role:
        # No new instructions but the Developer Role option is explicitly
        # set: convert the existing instructions message role to match it.
        # Build a new dict, the history messages are shared with the
        # (cached) upstream node output and must not be mutated.
        messages[0] = {**messages[0], "role": role}
    return messages


def _without_reasoning(messages: list[ChatCompletionMessageParam]) -> list[ChatCompletionMessageParam]:
    """The messages without their reasoning fields (see extract_reasoning),
    for the request only: the History output keeps them. New dicts for the
    messages that have some, the history ones are shared with the (cached)
    upstream node output and must not be mutated."""
    return [
        cast(ChatCompletionMessageParam, {k: v for k, v in msg.items() if k not in HISTORY_REASONING_KEYS})
        if any(key in msg for key in HISTORY_REASONING_KEYS) else msg
        for msg in messages
    ]


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
                    display_name="Image(s)",
                    optional=True,
                    tooltip="Image(s) to include in the request",
                ),
                # Optional: API format prompts saved before this input existed
                # have no value for it
                io.Boolean.Input(
                    id="send_reasoning",
                    display_name="Send Reasoning",
                    optional=True,
                    default=True,
                    advanced=True,
                    tooltip="Send the reasoning of the previous answers in the history (reasoning_content, reasoning or reasoning_details field with encrypted blocks from relays such as OpenRouter, as received) back to the server. Whether the model uses it is up to the server and its chat template; some keep it in the context, which grows it. Disable to send the history without it: the History output keeps it either way.",
                ),
                io.Boolean.Input(
                    id="show_reasoning",
                    display_name="Show Reasoning",
                    optional=True,
                    default=False,
                    advanced=True,
                    tooltip="Print the reasoning of the answer in the ComfyUI console (and the number of its encrypted blocks, never their content). Off by default: a long reasoning floods the console. The History output keeps it either way.",
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
                      send_reasoning: bool | None = True,
                      show_reasoning: bool | None = False,
                ) -> io.NodeOutput:
        # Linked inputs are not checked by validate_inputs (None there)
        if not isinstance(prompt, str) or prompt == "":
            raise ValueError(
                "Chat Completion: the prompt is empty (or not a text). If it comes from a "
                "Load Memory node, connect its Default input for the first run.")
        if not isinstance(model, str) or model == "":
            raise ValueError(
                "Chat Completion: the model is empty (or not a text). If it comes from a "
                "Load Memory node, connect its Default input for the first run.")
        check_input(system_prompt, str, "System Prompt", "a text", "Chat Completion")
        check_history(history, "Chat Completion")
        check_options(options, "Chat Completion")
        check_client(client, "Chat Completion")
        check_input(send_reasoning, bool, "Send Reasoning", "a boolean", "Chat Completion")
        check_input(show_reasoning, bool, "Show Reasoning", "a boolean", "Chat Completion")
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
            # None: a linked value with nothing stored, the default applies
            "messages": messages if send_reasoning is not False else _without_reasoning(messages),
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
        except openai.APIError as err:
            # The chained original exceptions are printed in the ComfyUI log:
            # strip any echoed credentials from them too
            scrub_exception(err, client)
            raise RuntimeError(f"Chat completion request failed: {describe_api_error(err, client)}") from err
        except Exception as err:
            # Not an SDK error (a Cancel, a bug below the SDK...): raised as
            # is, only without the credentials
            scrub_exception(err, client)
            raise
        response = extract_response_content(completion)
        # Add the response to the history, with its reasoning fields
        messages.append(_assistant_message(response))
        # Only on demand, a long reasoning floods the console (None: a linked
        # value with nothing stored, the default applies). Logged in full (no
        # node output shows it, only the History holds it), before the usage
        # stats.
        if show_reasoning:
            reasoning = reasoning_text(response.reasoning)
            if reasoning is not None:
                logger.info("Chat Completion: model reasoning:\n%s", reasoning)
            encrypted_blocks, encrypted_size = encrypted_reasoning(response.reasoning)
            if encrypted_blocks:
                logger.info("Chat Completion: %d encrypted reasoning block(s) (%d characters) "
                            "kept in the history", encrypted_blocks, encrypted_size)
        # Handle usage stats as text preview
        stats = format_usage(completion.usage, getattr(completion, "timings", None))
        # Also logged (shown in the console at the default INFO level), after
        # the log line of the HTTP request
        if stats:
            # On one line, easier to find in a busy console
            logger.info("Chat Completion usage: %s", stats.replace("\n", ", "))
        # Return the response and the history and the stats for the UI
        return io.NodeOutput(
            response.content,
            HistoryPayload(messages),
            ui=ui.PreviewText(stats or ""),
        )
