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


def scrub_exception(err: openai.APIError, client: AsyncOpenAI) -> None:
    """Strip the client API key from an SDK exception, in place, before it is
    chained to another one: ComfyUI prints the chained exception in its log."""
    err.args = tuple(redact(a, client) if isinstance(a, str) else a for a in err.args)
    if isinstance(getattr(err, "message", None), str):
        err.message = redact(err.message, client)


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
        hint = _STATUS_HINTS.get(status)
        if status in (404, 405):
            # Without any API prefix the route itself is probably wrong. With
            # one, a 404 usually means an unknown model (ollama, vLLM, OpenAI)
            if urlparse(str(client.base_url)).path in ("", "/"):
                hint = "the Base URL probably lacks the API prefix (usually /v1)"
            else:
                hint = "check the model name and the Base URL"
        if is_redirect:
            hint = "check the Base URL"
        if hints and status in hints:
            hint = hints[status]
        if hint is not None:
            message += f". Hint: {hint}"
        return redact(message, client)
    if isinstance(err, openai.APIConnectionError):
        reason = _connection_error_reason(err)
        kind = "timed out" if isinstance(err, openai.APITimeoutError) else "failed"
        detail = f"{type(reason).__name__}: {reason}" if str(reason) else type(reason).__name__
        return redact(f"cannot reach {client.base_url}: the connection {kind} ({detail})", client)
    return redact(str(err), client)
