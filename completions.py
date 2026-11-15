import base64
from io import BytesIO
import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

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
from .client import redact
from .interrupt import run_interruptible
from .iotypes import ParamClient, ParamHistory, ParamOptions, HistoryPayload, OptionsPayload, check_history


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


_STATUS_HINTS = {
    401: "authentication refused: check OPENAI_API_KEY (and that the server expects a Bearer token)",
    403: "authentication refused: check OPENAI_API_KEY (and that the server expects a Bearer token)",
}


def _response_body_snippet(response: Any, max_len: int = 300) -> str:
    try:
        text = response.text
    except Exception:
        return ""
    content_type = response.headers.get("content-type", "").lower()
    if "html" in content_type or text.lstrip()[:15].lower().startswith(("<!doctype html", "<html")):
        title = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        title_text = " ".join(title.group(1).split())[:100] if title else ""
        return f'HTML page "{title_text}"' if title_text else "HTML page"
    text = " ".join(text.split())
    return text[:max_len] + ("..." if len(text) > max_len else "")


def describe_api_error(err: Exception, client: AsyncOpenAI) -> str:
    """A clear error message for a failed request: method, URL, status, an
    excerpt of the answer and a hint. Never includes the request headers."""
    if isinstance(err, openai.APIStatusError):
        request = err.response.request
        status = err.status_code
        message = f"the server answered HTTP {status} to {request.method} {request.url}"
        snippet = _response_body_snippet(err.response)
        if snippet:
            message += f" ({snippet})"
        hint = _STATUS_HINTS.get(status)
        if status in (404, 405):
            # Without any API prefix the route itself is probably wrong. With
            # one, a 404 usually means an unknown model (ollama, vLLM, OpenAI)
            if urlparse(str(client.base_url)).path in ("", "/"):
                hint = "the Base URL probably lacks the API prefix (usually /v1)"
            else:
                hint = "check the model name and the Base URL"
        if hint is not None:
            message += f". Hint: {hint}"
        return redact(message, client)
    if isinstance(err, openai.APIConnectionError):
        reason = err.__cause__ or err.__context__ or err
        kind = "timed out" if isinstance(err, openai.APITimeoutError) else "failed"
        return redact(f"cannot reach {client.base_url}: the connection {kind} ({type(reason).__name__}: {reason})", client)
    return redact(str(err), client)


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
        if "force_regen" in kwargs and kwargs["force_regen"] is None:
            return str(time.time())  # linked input: always refresh
        if kwargs.get("force_regen"):
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
        check_history(history, "Chat Completion")
        # Handle options
        seed: int | None = None
        temperature: float | None = None
        max_tokens: int | None = None
        max_completion_tokens: int | None = None
        top_p: float | None = None
        frequency_penalty: float | None = None
        presence_penalty: float | None = None
        # A string: "none" is a real value that must be sent, only a Python
        # None (no Reasoning Effort node) omits the field
        reasoning_effort: str | None = None
        use_developer_role: bool = False
        developer_role_option_set: bool = False
        extra_body: dict[str, Any] = {}
        if options is not None:
            extra_body = options.get_options_copy()
            if "seed" in extra_body:
                seed = extra_body["seed"]
                del extra_body["seed"]
            if "temperature" in extra_body:
                temperature = extra_body["temperature"]
                del extra_body["temperature"]
            if "max_tokens" in extra_body:
                max_tokens = extra_body["max_tokens"]
                del extra_body["max_tokens"]
            if "max_completion_tokens" in extra_body:
                max_completion_tokens = extra_body["max_completion_tokens"]
                del extra_body["max_completion_tokens"]
            if "top_p" in extra_body:
                top_p = extra_body["top_p"]
                del extra_body["top_p"]
            if "frequency_penalty" in extra_body:
                frequency_penalty = extra_body["frequency_penalty"]
                del extra_body["frequency_penalty"]
            if "presence_penalty" in extra_body:
                presence_penalty = extra_body["presence_penalty"]
                del extra_body["presence_penalty"]
            if "reasoning_effort" in extra_body:
                reasoning_effort = extra_body["reasoning_effort"]
                del extra_body["reasoning_effort"]
            if "use_developer_role" in extra_body:
                developer_role_option_set = True
                use_developer_role = extra_body["use_developer_role"]
                del extra_body["use_developer_role"]
        # Handle system prompt
        if history is not None:
            messages = history.get_msgs_copy()
            first_msg_role = messages[0].get('role') if messages else None
            has_instructions_msg = first_msg_role in ("system", "developer")
            if system_prompt is not None and system_prompt != "":
                # Should we insert it at the beginning or replace the existing system message?
                if has_instructions_msg:
                    # Replace the existing system message
                    if use_developer_role:
                        messages[0] = {
                            "role": "developer",  # need literal for type hint check
                            "content": system_prompt,
                        }
                    else:
                        messages[0] = {
                            "role": "system",  # need literal for type hint check
                            "content": system_prompt,
                        }
                else:
                    # insert a new system/dev message at the begining of the list
                    if use_developer_role:
                        messages.insert(0, {
                            "role": "developer",  # need literal for type hint check
                            "content": system_prompt,
                        })
                    else:
                        messages.insert(0, {
                            "role": "system",  # need literal for type hint check
                            "content": system_prompt,
                        })
            elif has_instructions_msg and developer_role_option_set:
                # No new instructions but the Developer Role option is explicitly
                # set: convert the existing instructions message role to match it.
                # Build a new dict, the history messages are shared with the
                # (cached) upstream node output and must not be mutated.
                wanted_role = "developer" if use_developer_role else "system"
                if first_msg_role != wanted_role:
                    messages[0] = {**messages[0], "role": wanted_role}
        else:
            messages: list[ChatCompletionMessageParam] = []
            if system_prompt:
                if use_developer_role:
                    messages.append({
                        "role": "developer",  # need literal for type hint check
                        "content": system_prompt,
                    })
                else:
                    messages.append({
                        "role": "system",  # need literal for type hint check
                        "content": system_prompt,
                    })
        # Handle user message
        if images is not None:
            # Build multi modal content: every image of the batch, then the
            # prompt once (it must not be repeated for each image)
            content: list[ChatCompletionContentPartParam] = []
            for image in images:  # iterating a [B,H,W,C] tensor yields [H,W,C] images
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": comfy_image_to_base64_png_url(image)
                        }
                    }
                )
            content.append(
                {
                    "type": "text",
                    "text": prompt
                }
            )
            # Add the multi-modal content to the messages list
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )
        # Create the completion. Omit unset optional fields so providers are not
        # sent null values, and never send both token limits at once.
        if max_tokens is not None and max_completion_tokens is not None:
            raise ValueError(
                "max_tokens and max_completion_tokens cannot both be set. "
                "Use only one: max_tokens for older models and providers, "
                "or max_completion_tokens for newer OpenAI chat models."
            )
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "n": 1,
        }
        if seed is not None:
            request_kwargs["seed"] = seed  # deprecated, should we remove it?
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if max_completion_tokens is not None:
            request_kwargs["max_completion_tokens"] = max_completion_tokens
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if frequency_penalty is not None:
            request_kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            request_kwargs["presence_penalty"] = presence_penalty
        if reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = reasoning_effort
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        try:
            # Sent from the API client loop, where the client lives: its
            # connections are shared with the other nodes using this client
            completion = await run_interruptible(
                aioloop.submit(client.chat.completions.create(**request_kwargs)))
        except (openai.APIStatusError, openai.APIConnectionError) as err:
            # The chained original exception is printed in the ComfyUI log:
            # strip any echoed credentials from it too
            err.args = tuple(redact(a, client) if isinstance(a, str) else a for a in err.args)
            if isinstance(getattr(err, "message", None), str):
                err.message = redact(err.message, client)
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
        # add it to the console following the openai http call log for now as previewtext does not work yet
        if stats:
            print(stats)
        # Return the response and the history and the stats for the UI
        return io.NodeOutput(
            content,
            HistoryPayload(messages),
            ui=ui.PreviewText(stats) if stats else ui.PreviewText(""),
        )
