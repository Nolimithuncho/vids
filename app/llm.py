"""The AI layer. Two providers behind the same two functions, `parse` and `ask`:

- gemini (default): Google Gemini via the google-genai SDK. Has a free tier.
- claude: Anthropic Claude via the anthropic SDK. Pay-as-you-go API.

Messages come in as a simple shared format: a list of {"role": "user"|"assistant", "content": ...}
where content is a string or a list of {"type": "text", "text": ...} /
{"type": "image", "source": {"media_type": ..., "data": <base64>}} blocks.
"""
from __future__ import annotations

import base64
import re
import threading
import time
from typing import Callable
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)


_status_hook: Callable[[str | None], None] | None = None


def set_status_hook(hook: Callable[[str | None], None] | None) -> None:
    """Lets the pipeline show messages like "waiting 40s for the rate limit" in the UI."""
    global _status_hook
    _status_hook = hook


def _report(message: str | None) -> None:
    if message:
        print(message, flush=True)
    if _status_hook:
        try:
            _status_hook(message)
        except Exception:
            pass


class RefusalError(RuntimeError):
    pass


class AIError(RuntimeError):
    pass


def parse(system: str, messages: list[dict], out: Type[T], max_tokens: int = 32000) -> T:
    """One call that must return JSON matching `out`."""
    if config.PROVIDER == "claude":
        return _claude_parse(system, messages, out, max_tokens)
    return _gemini_parse(system, messages, out, max_tokens)


def ask(context: str, history: list[dict], question: str) -> str:
    """Answer a question about a video, given its notes as `context` and the chat so far."""
    if config.PROVIDER == "claude":
        return _claude_ask(context, history, question)
    return _gemini_ask(context, history, question)


QA_SYSTEM = """You are answering questions about a video that has already been watched and analyzed.
Below is everything known about it: the final report, detailed notes for every chunk (what was on screen,
steps, scenes, moments, text), and the full timestamped transcript. Answer from this material only. Cite
timestamps like [12:34] so the user can jump there. If the material doesn't contain the answer (for example
a visual detail between sampled frames), say so plainly rather than guessing. Keep answers direct."""


# ===================================================================== Gemini

_gclient = None
_glock = threading.Lock()


def _gemini():
    # One shared client. The lock matters: chunks run in parallel threads, and a second client
    # created by a race would replace the first, which then gets garbage-collected and closed
    # while a request is still using it.
    global _gclient
    with _glock:
        return _gemini_locked()


def _gemini_locked():
    global _gclient
    if _gclient is None:
        from google import genai
        from google.genai import types

        if not config.GEMINI_API_KEY:
            raise AIError("No Gemini API key. Put GEMINI_API_KEY=... in the .env file (free key: aistudio.google.com/apikey).")
        _gclient = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=types.HttpOptions(
                timeout=600_000,  # milliseconds
                # Retries are done in _gemini_call so the wait can be shown to the user.
                retry_options=types.HttpRetryOptions(attempts=1),
                **({"base_url": config.GEMINI_BASE_URL} if config.GEMINI_BASE_URL else {}),
            ),
        )
    return _gclient


def _gemini_contents(messages: list[dict]) -> list:
    from google.genai import types

    contents = []
    for m in messages:
        blocks = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        parts = []
        for b in blocks:
            if b["type"] == "text":
                parts.append(types.Part.from_text(text=b["text"]))
            elif b["type"] == "image":
                parts.append(types.Part.from_bytes(
                    data=base64.standard_b64decode(b["source"]["data"]), mime_type=b["source"]["media_type"],
                ))
        contents.append(types.Content(role="model" if m["role"] == "assistant" else "user", parts=parts))
    return contents


def _gemini_text(response: Any) -> str:
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        raise RefusalError(f"Gemini blocked this request ({feedback.block_reason}).")
    if not response.candidates:
        raise AIError("Gemini returned no answer.")
    reason = str(response.candidates[0].finish_reason or "")
    if "MAX_TOKENS" in reason:
        raise AIError("Gemini ran out of output space; try a shorter VID_SEGMENT_SECONDS.")
    text = response.text
    if not text:
        raise RefusalError(f"Gemini returned no text (finish reason: {reason or 'unknown'}).")
    return text


def _retry_delay(err: Exception, attempt: int) -> float:
    """Gemini's 429 errors say how long to wait ("retryDelay": "37s"); otherwise back off."""
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(err))
    if m:
        return min(120.0, float(m.group(1)) + 2)
    return min(90.0, 10.0 * (2 ** attempt))


def _gemini_call(system: str, messages: list[dict], max_tokens: int, schema: Type[BaseModel] | None) -> str:
    from google.genai import errors, types

    cfg = types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens)
    if schema is not None:
        cfg.response_mime_type = "application/json"
        cfg.response_schema = schema
    contents = _gemini_contents(messages)
    attempts = 10
    for attempt in range(attempts):
        try:
            response = _gemini().models.generate_content(model=config.GEMINI_MODEL, contents=contents, config=cfg)
            return _gemini_text(response)
        except errors.APIError as e:
            text = str(e)
            if e.code in (400, 401, 403) and "API key" in text:
                raise AIError("Gemini rejected the API key. Check GEMINI_API_KEY in your .env file.") from e
            if e.code == 404:
                raise AIError(f"Gemini doesn't know the model '{config.GEMINI_MODEL}'. Set VID_GEMINI_MODEL in .env "
                              "to a current model name (see ai.google.dev/gemini-api/docs/models).") from e
            if e.code == 429 and "PerDay" in text:
                raise AIError("Gemini's free daily limit is used up. Press Retry tomorrow; finished parts are kept.") from e
            if e.code not in (408, 429, 500, 502, 503, 504) or attempt == attempts - 1:
                if e.code == 429:
                    raise AIError("Gemini's free-tier limit was reached. Wait a while and press Retry; "
                                  "finished parts are kept.") from e
                raise AIError(f"Gemini error {e.code}: {e}") from e
            delay = _retry_delay(e, attempt)
            why = "free-tier limit (requests per minute)" if e.code == 429 else f"Gemini is busy (error {e.code})"
            _report(f"Waiting {delay:.0f}s: {why}. It will continue by itself.")
            time.sleep(delay)
            _report(None)
        except (AIError, RefusalError):
            raise
        except Exception as e:  # network trouble (no internet, connection reset, timeout)
            if attempt >= 4:
                raise AIError(f"Couldn't reach Gemini ({type(e).__name__}: {e}). Check your internet and press Retry.") from e
            delay = 10.0 * (attempt + 1)
            _report(f"Waiting {delay:.0f}s: couldn't reach Gemini ({type(e).__name__}). It will try again by itself.")
            time.sleep(delay)
            _report(None)
    raise AIError("Gemini did not answer.")


def _gemini_parse(system: str, messages: list[dict], out: Type[T], max_tokens: int) -> T:
    return out.model_validate_json(_gemini_call(system, messages, max_tokens, out))


def _gemini_ask(context: str, history: list[dict], question: str) -> str:
    system = QA_SYSTEM + "\n\n" + context
    return _gemini_call(system, list(history) + [{"role": "user", "content": question}], 16000, None)


# ===================================================================== Claude

_aclient = None


def _claude():
    global _aclient
    if _aclient is None:
        import anthropic

        # Reads ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment.
        _aclient = anthropic.Anthropic(max_retries=4, timeout=600)
    return _aclient


def _claude_extra() -> dict:
    """Server-side refusal fallbacks: if the model declines a chunk, the API reruns it on
    Anthropic's recommended fallback model instead of failing the whole job."""
    if not config.USE_FALLBACKS:
        return {}
    return {
        "extra_headers": {"anthropic-beta": "server-side-fallback-2026-07-01"},
        "extra_body": {"fallbacks": "default"},
    }


def _strict_schema(model: Type[BaseModel]) -> dict:
    schema = model.model_json_schema()

    def fix(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("title", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}).keys())
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(schema)
    return schema


def _claude_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            content = [
                {"type": "image", "source": {"type": "base64", **b["source"]}} if b["type"] == "image" else b
                for b in content
            ]
        out.append({"role": m["role"], "content": content})
    return out


def _claude_text(message: Any) -> str:
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        raise RefusalError(f"The model declined this request ({getattr(details, 'category', None)}).")
    if message.stop_reason == "max_tokens":
        raise AIError("The model ran out of output tokens; try a shorter VID_SEGMENT_SECONDS.")
    return "".join(b.text for b in message.content if b.type == "text")


def _claude_parse(system: str, messages: list[dict], out: Type[T], max_tokens: int) -> T:
    # Streams so long outputs don't hit HTTP timeouts.
    with _claude().messages.stream(
        model=config.CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=_claude_messages(messages),
        output_config={
            "effort": config.EFFORT,
            "format": {"type": "json_schema", "schema": _strict_schema(out)},
        },
        **_claude_extra(),
    ) as stream:
        message = stream.get_final_message()
    return out.model_validate_json(_claude_text(message))


def _claude_ask(context: str, history: list[dict], question: str) -> str:
    with _claude().messages.stream(
        model=config.CLAUDE_MODEL,
        max_tokens=16000,
        # The big video context is identical across questions, so cache it.
        system=[
            {"type": "text", "text": QA_SYSTEM},
            {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
        ],
        messages=list(history) + [{"role": "user", "content": question}],
        output_config={"effort": config.EFFORT},
        **_claude_extra(),
    ) as stream:
        message = stream.get_final_message()
    return _claude_text(message)
