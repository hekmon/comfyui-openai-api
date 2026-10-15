import os
from urllib.parse import urlparse

from openai import OpenAI
from comfy_api.latest import io

from .iotypes import ParamClient

# API keys must never travel through widget inputs: ComfyUI serializes every
# widget value (even from bypassed nodes) into the `prompt` and `workflow`
# metadata chunks of generated images and exported workflows, which would leak
# them. The key is instead read from the OPENAI_API_KEY environment variable
# at execution time. The "-" placeholder keeps keyless inference servers
# (vLLM, etc) working: they require an Authorization header but do not
# validate its value.
_DEFAULT_KEYLESS_API_KEY = "-"


class Client(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_Client",
            display_name="OpenAI API - Client",
            category="OpenAI API",
            description="The OpenAI API client used to perform chat completions. The API key is read from the OPENAI_API_KEY environment variable at execution time and must never be stored in the workflow: ComfyUI embeds widget values into the metadata of generated images, which would leak it.",
            inputs=[
                io.String.Input(
                    id="base_url",
                    display_name="Base URL",
                    tooltip="The base URL to use for the OpenAI API requests",
                    placeholder="http(s)://host[:port][/URI]",
                    default="https://api.openai.com/v1"
                ),
                io.Int.Input(
                    id="max_retries",
                    display_name="Max Retries",
                    tooltip="Max number of retries for failed requests",
                    default=2,
                    min=0,
                ),
                io.Int.Input(
                    id="timeout",
                    display_name="Timeout",
                    tooltip="Request timeout in seconds",
                    default=600,
                    min=1,
                ),
            ],
            outputs=[
                ParamClient.Output(
                    id="client",
                    display_name="API Client",
                    tooltip="The initialized and ready to query OpenAI API client"
                )
            ],
        )

    @classmethod
    def validate_inputs(cls, base_url: str | None) -> bool | str:
        # base_url can be None if coming from another node
        if base_url is not None:
            try:
                result = urlparse(base_url)
                if result.scheme not in ["http", "https"]:
                    return "URL scheme must be http or https"
            except ValueError as e:
                return f"invalid URL: {e}"
        return True

    @classmethod
    def execute(cls, base_url: str, max_retries: int, timeout: int) -> io.NodeOutput:
        return io.NodeOutput(
            OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", _DEFAULT_KEYLESS_API_KEY),
                base_url=base_url,
                max_retries=max_retries,
                timeout=timeout
            )
        )
