import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import comfy.model_management
import openai
from openai import AsyncOpenAI

from comfy_api.latest import io

from . import aioloop
from .apierrors import describe_api_error, scrub_exception
from .interrupt import run_interruptible
from .iotypes import ParamClient, check_client, check_input


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ServerType:
    """How a backend unloads a model: a POST to one of its native routes."""
    path: str  # relative to the server root (see server_root)
    needs_model: bool
    # The request body for a model name, None for no body at all
    body: Callable[[str], dict[str, Any] | None]
    # Appended to the default hints of the 401/403 and 404/405 errors
    auth_hint: str = ""
    route_hint: str = ""


# Backend specific unload calls (the OpenAI API has no such endpoint). Only
# backends that load the model again on the next request are listed: an unload
# must never break the next generation. vLLM sleep mode and SGLang memory
# release are left out for that reason (requests wait or fail until an explicit
# wake up / resume call). The keys are the Server Type options, in this order.
SERVER_TYPES: dict[str, ServerType] = {
    "tabbyAPI": ServerType(
        "/v1/model/unload", needs_model=False,
        body=lambda model: None,  # unloads the loaded model, no body
        auth_hint=" (tabbyAPI requires an admin key to unload)"),
    "llama.cpp (router mode)": ServerType(
        "/models/unload", needs_model=True,
        body=lambda model: {"model": model},
        route_hint=" (llama.cpp only has this route in router mode)"),
    "LM Studio": ServerType(
        "/api/v1/models/unload", needs_model=True,
        body=lambda model: {"instance_id": model}),
    "ollama": ServerType(
        "/api/generate", needs_model=True,
        # an empty prompt with keep_alive 0 unloads the model
        body=lambda model: {"model": model, "keep_alive": 0}),
}


def server_root(base_url: str) -> str:
    """The OpenAI-compatible base URL usually ends with /v1 while the native
    unload routes live at the server root: strip it."""
    return base_url.rstrip("/").removesuffix("/v1")


def check_unload_inputs(server_type: str, model: str) -> str | None:
    """The error message for an unknown server type or a missing model (model
    already stripped), None if valid."""
    if server_type not in SERVER_TYPES:
        return f"unknown server type {server_type!r} (expected one of: {', '.join(SERVER_TYPES)})"
    if SERVER_TYPES[server_type].needs_model and model == "":
        return f"model is required to unload from {server_type}"
    return None


def _unload_hints(server_type: str) -> dict[int, str]:
    """Hints for the HTTP errors of an unload call (see describe_api_error)."""
    spec = SERVER_TYPES[server_type]
    auth = "authentication refused: check OPENAI_API_KEY" + spec.auth_hint
    route = f"check the Server Type ({server_type}), the model name and the Base URL" + spec.route_hint
    return {401: auth, 403: auth, 404: route, 405: route}


async def unload_model(client: AsyncOpenAI, server_type: str, model: str, timeout: float) -> None:
    """Ask a local inference server to unload its model. Raises on failure.
    Based on the unload_localhosted_models helper of PR #11 by nikich340.

    Sent by the client itself (same headers, proxies and connection pool as
    the chat completions: an open connection to the server is reused), on the
    API client loop where it lives, so that Cancel stops it at once."""
    spec = SERVER_TYPES[server_type]
    url = server_root(str(client.base_url)) + spec.path  # absolute: not joined to the base URL
    body = spec.body(model)
    options: dict[str, Any] = {
        "timeout": timeout,
        # An unload is never retried silently, whatever the client Max Retries
        "max_retries": 0,
        # Never follow redirects: the Authorization header (the API key) must
        # only go to the base URL host, and a redirected request could report
        # a success for an unload that never happened. A 3xx is an error.
        "follow_redirects": False,
    }
    if body is None:
        # No body (tabbyAPI): no JSON Content-Type either
        options["headers"] = {"Content-Type": openai.Omit()}
    # cast_to=str: the answer is only read as text, never parsed: unload routes
    # may answer with an empty body or plain text
    request = client.post(url, cast_to=str, body=body, options=options)  # type: ignore[arg-type]
    try:
        await run_interruptible(aioloop.submit(request))
    except openai.APIError as err:
        # Same as Chat Completion: the message never includes the request
        # headers and strips the key from the server answer (some servers
        # echo the credentials), as does scrub_exception from the chained
        # original exceptions, printed in the ComfyUI log.
        scrub_exception(err, client)
        if isinstance(err, openai.APITimeoutError):
            reason = f" (POST {url}): no answer within {timeout} s"
        elif isinstance(err, openai.APIStatusError):
            # the description already has the method and URL
            reason = ": " + describe_api_error(err, client, hints=_unload_hints(server_type))
        else:
            reason = f" (POST {url}): " + describe_api_error(err, client, hints=_unload_hints(server_type))
        raise RuntimeError(f"{server_type} unload failed{reason}") from err
    except Exception as err:
        # Not an SDK error (a Cancel, a bug below the SDK...): raised as is,
        # only without the credentials
        scrub_exception(err, client)
        raise


class UnloadModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_UnloadModel",
            display_name="OpenAI API - Unload Model",
            category="OpenAI API",
            description="Asks a local inference server to unload its model to free VRAM, e.g. before an image generation. Runs after its Value input is ready (connect the chat completion output) and passes the value through. The server loads the model again on the next request (see the server type tooltip for the needed settings).",
            is_output_node=True,
            inputs=[
                io.AnyType.Input(
                    id="value",
                    display_name="Value",
                    tooltip="Any value, passed through once the model is unloaded. Connect the output of the node that must run before the unload (e.g. the chat completion response).",
                ),
                ParamClient.Input(
                    id="client",
                    display_name="API Client",
                    tooltip="Same Client as your Chat Completion node, no change needed (keep the usual .../v1 base URL). Its API key is reused, and the server root for native unload routes is derived from the base URL automatically.",
                ),
                io.Combo.Input(
                    id="server_type",
                    display_name="Server Type",
                    options=list(SERVER_TYPES.keys()),
                    default="ollama",
                    tooltip="The inference server software. tabbyAPI: needs an admin key in OPENAI_API_KEY, and reloads on the next request only with inline_model_loading: true in its config.yml (and an admin key). llama.cpp: only in router mode (llama-server started without -m), reloads on demand unless --no-models-autoload. LM Studio: reloads on demand with JIT loading enabled (default). ollama: reloads on demand.",
                ),
                io.String.Input(
                    id="model",
                    display_name="Model",
                    tooltip="The model to unload, same name as in the chat completion node. Required for llama.cpp, LM Studio and ollama, ignored by tabbyAPI (unloads its only model).",
                    default="",
                ),
                io.Float.Input(
                    id="timeout",
                    display_name="Timeout",
                    tooltip="Unload request timeout in seconds",
                    default=10.0,
                    min=0.5,
                    max=300.0,
                    step=0.5,
                    display_mode=io.NumberDisplay.number,
                    advanced=True,
                ),
                io.Boolean.Input(
                    id="fail_on_error",
                    display_name="Fail On Error",
                    tooltip="If false, an unload failure only logs a warning and the value is still passed through. If true, it stops the workflow with an error.",
                    default=False,
                    advanced=True,
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    id="value_out",
                    display_name="Value",
                    tooltip="The input value, passed through after the unload request",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, server_type: str | None, model: str | None) -> bool | str:
        # None when coming from another node: checked at execution. Listing
        # server_type here disables the ComfyUI check of the Combo options:
        # check_unload_inputs rejects an unknown server type instead.
        if server_type is None or model is None:
            return True
        return check_unload_inputs(server_type, model.strip()) or True

    @classmethod
    async def execute(cls,
                      value: Any,
                      client: AsyncOpenAI,
                      server_type: str,
                      model: str,
                      timeout: float,
                      fail_on_error: bool,
                ) -> io.NodeOutput:
        try:
            check_client(client, "Unload Model")
            # A linked model can be None (nothing stored): no model, which
            # tabbyAPI does not need
            check_input(model, str, "Model", "a text", "Unload Model")
            model = (model or "").strip()
            error = check_unload_inputs(server_type, model)
            if error is not None:
                raise ValueError(error)
            await unload_model(client, server_type, model, timeout)
            logger.info("%s: model %s unloaded", server_type, model or "(current)")
        except comfy.model_management.InterruptProcessingException:
            # An Exception subclass in older ComfyUI versions: never turn a
            # Cancel into a warning
            raise
        except Exception as e:
            if fail_on_error:
                raise
            logger.warning("%s (value passed through anyway)", e)
        return io.NodeOutput(value)
