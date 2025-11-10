from urllib.parse import urlparse

from openai import OpenAI
from comfy_api.latest import io

from .iotypes import ParamClient


class Client(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_Client",
            display_name="OpenAI API - Client",
            category="OpenAI API",
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
                    tooltip="The base URL to use for the OpenAI API requests",
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
                io.String.Input(
                    id="api_key",
                    display_name="API Key",
                    optional=True,
                    tooltip="The API key to use, if any",
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
    def execute(cls, base_url: str, max_retries: int, timeout: int, api_key: str | None = None) -> io.NodeOutput:
        return io.NodeOutput(
            OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=max_retries,
                timeout=timeout
            )
        )
