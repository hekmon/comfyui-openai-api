import json
import logging
import urllib.error
import urllib.request
from typing import Any

from openai import OpenAI

from comfy_api.latest import io

from .client import redact
from .iotypes import ParamClient


logger = logging.getLogger(__name__)

# Backend specific unload calls (the OpenAI API has no such endpoint). Only
# backends that load the model again on the next request are listed: an unload
# must never break the next generation. vLLM sleep mode and SGLang memory
# release are left out for that reason (requests wait or fail until an explicit
# wake up / resume call).
# Values: (method, path relative to the server root, needs a model name)
SERVER_TYPES: dict[str, tuple[str, str, bool]] = {
    "tabbyAPI": ("POST", "/v1/model/unload", False),
    "llama.cpp (router mode)": ("POST", "/models/unload", True),
    "LM Studio": ("POST", "/api/v1/models/unload", True),
    "ollama": ("POST", "/api/generate", True),
}


def server_root(base_url: str) -> str:
    """The OpenAI-compatible base URL usually ends with /v1 while the native
    unload routes live at the server root: strip it."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-len("/v1")]
    return root


def unload_request_body(server_type: str, model: str) -> dict[str, Any] | None:
    if server_type == "llama.cpp (router mode)":
        return {"model": model}
    if server_type == "LM Studio":
        return {"instance_id": model}
    if server_type == "ollama":
        # an empty prompt with keep_alive 0 unloads the model
        return {"model": model, "keep_alive": 0}
    return None  # tabbyAPI: unloads the loaded model, no body


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow redirects: urllib would forward the Authorization header
    (the API key) to any host, turn the POST into a GET and could report a
    success for an unload that never happened. A 3xx is raised as an error."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def unload_model(client: OpenAI, server_type: str, model: str, timeout: float) -> None:
    """Ask a local inference server to unload its model. Raises on failure.
    Based on the unload_localhosted_models helper of PR #11 by nikich340."""
    method, path, _ = SERVER_TYPES[server_type]
    url = server_root(str(client.base_url)) + path
    headers = {"Authorization": f"Bearer {client.api_key}"}
    body = unload_request_body(server_type, model)
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            status = response.status
    except urllib.error.HTTPError as e:
        # Only the start of the server answer: never the request headers. The
        # key is still stripped from it, some servers echo the credentials.
        detail = e.read(300).decode("utf-8", errors="replace").strip()
        if 300 <= e.code < 400:
            detail = f"redirect to {e.headers.get('Location', '?')} refused, check the base URL"
        raise RuntimeError(redact(f"{server_type} unload failed ({method} {url}): HTTP {e.code} {detail}", client)) from None
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise RuntimeError(redact(f"{server_type} unload failed ({method} {url}): {reason}", client)) from None
    if not 200 <= status < 300:
        raise RuntimeError(redact(f"{server_type} unload failed ({method} {url}): HTTP {status}", client))


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
                ),
                io.Boolean.Input(
                    id="fail_on_error",
                    display_name="Fail On Error",
                    tooltip="If false, an unload failure only logs a warning and the value is still passed through. If true, it stops the workflow with an error.",
                    default=False,
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
        # None when coming from another node: checked at execution
        if server_type is not None and model is not None:
            if server_type in SERVER_TYPES and SERVER_TYPES[server_type][2] and model.strip() == "":
                return f"model is required to unload from {server_type}"
        return True

    @classmethod
    def execute(cls,
                value: Any,
                client: OpenAI,
                server_type: str,
                model: str,
                timeout: float,
                fail_on_error: bool,
                ) -> io.NodeOutput:
        model = (model or "").strip()
        try:
            if SERVER_TYPES[server_type][2] and model == "":
                raise ValueError(f"model is required to unload from {server_type}")
            unload_model(client, server_type, model, timeout)
            logger.info("%s: model %s unloaded", server_type, model or "(current)")
        except Exception as e:
            if fail_on_error:
                raise
            logger.warning("%s (value passed through anyway)", e)
        return io.NodeOutput(value)
