import json
import logging
import urllib.error
import urllib.request

from openai import OpenAI
from comfy_api.latest import io

from .iotypes import UnloadableResponsePayload, ParamUnloadableResponse


_UNLOAD_TIMEOUT_SEC = 5.0

logger = logging.getLogger(__name__)


# Unload endpoints and strategy based on the research of @nikich340 in #11.
def unload_model(client: OpenAI, model: str, backend: str, timeout_sec: float = _UNLOAD_TIMEOUT_SEC) -> str:
    base_url = str(client.base_url).rstrip("/")
    if base_url.endswith("/v1"):
        # Ollama endpoints live at the server root, not under /v1
        base_url = base_url[:-len("/v1")]
    auth_header = {
        "Authorization": f"Bearer {client.api_key}",
    }
    json_header = {
        "Content-Type": "application/json",
    }
    try:
        if backend == "TabbyAPI":
            request = urllib.request.Request(
                base_url + "/v1/model/unload",
                method="POST",
                headers=auth_header,
            )
        elif backend == "LM Studio":
            body = json.dumps({"instance_id": model}).encode("utf-8")
            headers = {**auth_header, **json_header}
            request = urllib.request.Request(
                base_url + "/api/v1/models/unload",
                data=body,
                method="POST",
                headers=headers,
            )
        elif backend == "llama.cpp (router)":
            body = json.dumps({"model": model}).encode("utf-8")
            headers = {**auth_header, **json_header}
            request = urllib.request.Request(
                base_url + "/models/unload",
                data=body,
                method="POST",
                headers=headers,
            )
        elif backend == "Ollama":
            body = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
            request = urllib.request.Request(
                base_url + "/api/generate",
                data=body,
                method="POST",
                headers=json_header,
            )
        elif backend == "vLLM":
            request = urllib.request.Request(
                base_url + "/sleep?level=2",
                method="POST",
                headers=auth_header,
            )
        else:
            return f"failed (ValueError: unknown backend {backend})"
        response = None
        try:
            response = urllib.request.urlopen(request, timeout=timeout_sec)
            if response.status == 200 or response.status == 204:
                return f"ok (HTTP {response.status})"
            return f"failed (HTTP {response.status})"
        finally:
            if response is not None:
                response.close()
    except urllib.error.HTTPError as exc:
        # urlopen raises for non 2xx statuses instead of returning a response
        return f"failed (HTTP {exc.code})"
    except TimeoutError:
        return "failed (timeout)"
    except Exception as exc:
        return f"failed ({type(exc).__name__}: {exc})"


class UnloadModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_UnloadModel",
            display_name="OpenAI API - Unload Model",
            category="OpenAI API",
            description="Unloads the model from a local inference server after the response has been produced, freeing VRAM. The unload attempt is best effort and never fails the workflow.",
            inputs=[
                ParamUnloadableResponse.Input(
                    id="unloadable_response",
                    display_name="Unloadable Response",
                    tooltip="Connect the Unloadable Response output of the Chat Completion node",
                ),
                io.Combo.Input(
                    id="backend",
                    display_name="Backend",
                    options=["TabbyAPI", "LM Studio", "llama.cpp (router)", "Ollama", "vLLM"],
                    default="Ollama",
                    tooltip="Which local inference server to unload. vLLM requires launching the server with VLLM_SERVER_DEV_MODE=1 and --enable-sleep-mode. Targeting the official OpenAI API makes this node a no-op.",
                ),
            ],
            outputs=[
                io.String.Output(
                    id="response",
                    display_name="Response",
                    tooltip="The chat completion response, forwarded unchanged after the unload attempt",
                ),
            ],
        )

    @classmethod
    def execute(cls, unloadable_response: UnloadableResponsePayload, backend: str) -> io.NodeOutput:
        try:
            outcome = unload_model(unloadable_response.client, unloadable_response.model, backend)
            logger.info("Unload attempt on backend %s for model %s: %s", backend, unloadable_response.model, outcome)
        except Exception as exc:
            outcome = f"failed ({type(exc).__name__}: {exc})"
            logger.warning("Unload attempt on backend %s for model %s: %s", backend, unloadable_response.model, outcome)
        return io.NodeOutput(unloadable_response.response)
