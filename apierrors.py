import contextlib
import re
from typing import Any
from urllib.parse import urlparse

import openai
from openai import AsyncOpenAI

_STATUS_HINTS = {
    401: "authentication refused: check OPENAI_API_KEY (and that the server expects a Bearer token)",
    403: "authentication refused: check OPENAI_API_KEY (and that the server expects a Bearer token)",
}


def redact(text: str, client: AsyncOpenAI) -> str:
    """Strip the client API key from a text (error message, server answer...):
    some servers and proxies echo the received credentials in their errors."""
    api_key = getattr(client, "api_key", None)
    if isinstance(api_key, str) and len(api_key) > 3:
        text = text.replace(api_key, "***")
    return text


def _redact_value(value: Any, client: AsyncOpenAI) -> Any:
    # Texts, also inside the parsed JSON of an error body
    if isinstance(value, str):
        return redact(value, client)
    if isinstance(value, dict):
        return {k: _redact_value(v, client) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v, client) for v in value]
    return value


# Exception attributes that hold texts: the SDK message and parsed error
# body, the OSError fields shown by str()
_SCRUBBED_ATTRIBUTES = ("message", "body", "strerror", "filename", "filename2")


def scrub_exception(err: BaseException, client: AsyncOpenAI) -> None:
    """Strip the client API key, in place, from an exception and from every
    exception it is chained to (__cause__ and __context__, e.g. the HTTP
    client error below an SDK one). Done before raising it again or chaining
    it to the node error: ComfyUI prints the whole chain in its log. The raw
    HTTP request and response objects are left as they are: never printed."""
    seen: set[int] = set()
    pending: list[BaseException | None] = [err]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        current.args = tuple(_redact_value(a, client) for a in current.args)
        for attr in _SCRUBBED_ATTRIBUTES:
            value = getattr(current, attr, None)
            redacted = _redact_value(value, client)
            if redacted != value:
                with contextlib.suppress(AttributeError, TypeError):  # read-only
                    setattr(current, attr, redacted)
        pending += [current.__cause__, current.__context__]


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


def _connection_error_reason(err: BaseException) -> BaseException:
    """The most telling cause of a connection error. The async HTTP client
    wraps the OS error (e.g. "[Errno 111] Connect call failed") in generic
    ones ("All connection attempts failed"): prefer it when there is one."""
    first = err.__cause__ or err.__context__ or err
    seen: set[int] = set()
    current: BaseException | None = first
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno is not None:
            return current
        current = current.__cause__ or current.__context__
    return first


def describe_api_error(err: Exception, client: AsyncOpenAI, hints: dict[int, str] | None = None) -> str:
    """A clear error message for a failed request: method, URL, status, an
    excerpt of the answer and a hint (hints: replacements for the default
    ones, by HTTP status). Never includes the request headers, and strips the
    API key (see redact)."""
    if isinstance(err, openai.APIStatusError):
        request = err.response.request
        status = err.status_code
        message = f"the server answered HTTP {status} to {request.method} {request.url}"
        is_redirect = 300 <= status < 400
        if is_redirect:
            # Only for requests sent without following redirects
            message += f", a redirect to {err.response.headers.get('location', '?')} that was not followed"
        snippet = _response_body_snippet(err.response)
        if snippet:
            message += f" ({snippet})"
        # The first matching hint wins
        if hints and status in hints:
            hint = hints[status]
        elif is_redirect:
            hint = "check the Base URL"
        elif status in (404, 405):
            # Without any API prefix the route itself is probably wrong. With
            # one, a 404 usually means an unknown model (ollama, vLLM, OpenAI)
            if urlparse(str(client.base_url)).path in ("", "/"):
                hint = "the Base URL probably lacks the API prefix (usually /v1)"
            else:
                hint = "check the model name and the Base URL"
        else:
            hint = _STATUS_HINTS.get(status)
        if hint is not None:
            message += f". Hint: {hint}"
        return redact(message, client)
    if isinstance(err, openai.APIConnectionError):
        reason = _connection_error_reason(err)
        kind = "timed out" if isinstance(err, openai.APITimeoutError) else "failed"
        detail = f"{type(reason).__name__}: {reason}" if str(reason) else type(reason).__name__
        return redact(f"cannot reach {client.base_url}: the connection {kind} ({detail})", client)
    return redact(str(err), client)
