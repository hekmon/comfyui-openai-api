import logging
from urllib.parse import urlparse

from openai import AsyncOpenAI, OpenAIError
from comfy_api.latest import io

from . import aioloop
from .apierrors import KEYLESS_API_KEY, has_no_api_prefix
from .iotypes import ParamClient

# API keys must never travel through widget inputs: ComfyUI serializes every
# widget value (even from bypassed nodes) into the `prompt` and `workflow`
# metadata chunks of generated images and exported workflows, which would leak
# them. The key instead comes from the OPENAI_API_KEY environment variable at
# execution time, with a placeholder for keyless servers (KEYLESS_API_KEY).
# The variable is read by the OpenAI SDK itself (its documented default when no
# api_key is given): this module only checks what the SDK picked up.

logger = logging.getLogger(__name__)

# Base URLs already warned about, to log the missing prefix warning only once
_warned_base_urls: set[str] = set()


def clean_api_key(api_key: str) -> str:
    """Clean the API key read by the SDK from the OPENAI_API_KEY environment
    variable, the keyless placeholder when it is empty. The key ends up in the
    Authorization HTTP header, which only accepts printable ASCII: fail early
    with a clear message (never showing the value) instead of an encoding
    error deep inside the HTTP client."""
    # Surrounding whitespace (e.g. a trailing newline from a copy/paste) is
    # never part of a key
    api_key = api_key.strip()
    if api_key == "":
        return KEYLESS_API_KEY
    invalid_at = next((i for i, c in enumerate(api_key) if not " " <= c <= "~"), None)
    if invalid_at is not None:
        raise ValueError(
            "The OPENAI_API_KEY environment variable does not contain a valid API key: "
            f"it is {len(api_key)} characters long and has a non ASCII or control character "
            f"at position {invalid_at}, which an HTTP header can not carry (its value is not shown on purpose). "
            "It probably holds pasted text instead of a key: check where it is set "
            "(system or user environment variables, or the environment variables of your "
            "ComfyUI Desktop installation) and restart ComfyUI."
        )
    return api_key


def check_base_url(base_url: str) -> str | None:
    """The error message for an unusable Base URL, None if valid. Never
    let an empty one through: the SDK would silently replace it with the
    OPENAI_BASE_URL environment variable or https://api.openai.com/v1."""
    if base_url.strip() == "":
        return "the Base URL is empty"
    try:
        scheme = urlparse(base_url).scheme
    except ValueError as e:
        return f"invalid URL: {e}"
    if scheme not in ("http", "https"):
        return "URL scheme must be http or https"
    return None


async def _build_client(base_url: str, max_retries: int, timeout: int) -> AsyncOpenAI:
    # A coroutine only to run on the API client loop
    options = {"base_url": base_url, "max_retries": max_retries, "timeout": timeout}
    try:
        # No api_key: the SDK reads the OPENAI_API_KEY environment variable
        client = AsyncOpenAI(**options)
    except OpenAIError:
        # Raised only when the variable is unset: keyless server
        return AsyncOpenAI(api_key=KEYLESS_API_KEY, **options)
    # Set but empty (the SDK would then send no Authorization header at all),
    # padded with whitespace or invalid: replaced, trimmed or rejected
    api_key = clean_api_key(client.api_key)
    if api_key != client.api_key:
        # Shares the HTTP client (and its connections) of the original one
        client = client.with_options(api_key=api_key)
    return client


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
                    tooltip="Base URL of the OpenAI-compatible API, including its path prefix (usually /v1): the part before /chat/completions, nothing is added automatically. Examples: http://localhost:8000/v1 (vLLM, SGLang, tabbyAPI...), http://localhost:11434/v1 (ollama), https://api.openai.com/v1. Some providers use another prefix, and llama.cpp also answers without /v1.",
                    placeholder="http(s)://host[:port][/URI]",
                    default="https://api.openai.com/v1"
                ),
                io.Int.Input(
                    id="max_retries",
                    display_name="Max Retries",
                    tooltip="Max number of retries for failed requests",
                    default=2,
                    min=0,
                    max=10,
                    advanced=True,
                ),
                io.Int.Input(
                    id="timeout",
                    display_name="Timeout",
                    tooltip="Request timeout in seconds",
                    default=600,
                    min=1,
                    max=86400,  # explicit, otherwise the frontend caps it at 2048
                    advanced=True,
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
        # None when coming from another node: checked at execution
        if base_url is None:
            return True
        return check_base_url(base_url) or True

    @classmethod
    async def execute(cls, base_url: str, max_retries: int, timeout: int) -> io.NodeOutput:
        # A linked Base URL is not checked by validate_inputs (None there)
        if not isinstance(base_url, str) or base_url.strip() == "":
            raise ValueError(
                "Client: the Base URL is empty (or not a text). If it comes from a "
                "Load Memory node, connect its Default input for the first run.")
        error = check_base_url(base_url)
        if error is not None:
            raise ValueError(f"Client: {error}")
        # No API prefix at all: fine for llama.cpp, a 404/405 for most other
        # servers. Only a warning (once per URL), never added automatically.
        if has_no_api_prefix(base_url) and base_url not in _warned_base_urls:
            _warned_base_urls.add(base_url)
            logger.warning(
                "OpenAI API client: the base URL %s has no API path prefix (usually /v1). "
                "This works with llama.cpp, most other servers expect .../v1", base_url)
        # One async client shared by every node connected to this one, and
        # reused across prompts while ComfyUI caches it: its connections are
        # kept alive between requests. Built and used on the API client loop
        # only (see aioloop), never on the per prompt ComfyUI one.
        client = await aioloop.submit(_build_client(base_url, max_retries, timeout))
        aioloop.close_when_collected(client)
        return io.NodeOutput(client)
