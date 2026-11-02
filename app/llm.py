"""Thin wrapper around the Anthropic SDK for structured analysis calls and Q&A."""
from __future__ import annotations

from typing import Any, Type, TypeVar

import anthropic
from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # Reads ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment.
        _client = anthropic.Anthropic(max_retries=4, timeout=600)
    return _client


class RefusalError(RuntimeError):
    pass


def _extra() -> dict:
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


def _text(message: Any) -> str:
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        raise RefusalError(f"The model declined this request ({getattr(details, 'category', None)}).")
    if message.stop_reason == "max_tokens":
        raise RuntimeError("The model ran out of output tokens; try a shorter segment length.")
    return "".join(b.text for b in message.content if b.type == "text")


def parse(system: str, messages: list[dict], out: Type[T], max_tokens: int = 32000) -> T:
    """One structured-output call. Streams so long outputs don't hit HTTP timeouts."""
    with client().messages.stream(
        model=config.MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        output_config={
            "effort": config.EFFORT,
            "format": {"type": "json_schema", "schema": _strict_schema(out)},
        },
        **_extra(),
    ) as stream:
        message = stream.get_final_message()
    return out.model_validate_json(_text(message))


QA_SYSTEM = """You are answering questions about a video that has already been watched and analyzed.
Below is everything known about it: the final report, detailed notes for every chunk (what was on screen,
scenes, moments, text), and the full timestamped transcript. Answer from this material only. Cite timestamps
like [12:34] so the user can jump there. If the material doesn't contain the answer (for example a visual
detail between sampled frames), say so plainly rather than guessing. Keep answers direct."""


def ask(context: str, history: list[dict], question: str) -> str:
    messages = list(history) + [{"role": "user", "content": question}]
    with client().messages.stream(
        model=config.MODEL,
        max_tokens=16000,
        # The big video context is identical across questions, so cache it.
        system=[
            {"type": "text", "text": QA_SYSTEM},
            {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
        ],
        messages=messages,
        output_config={"effort": config.EFFORT},
        **_extra(),
    ) as stream:
        message = stream.get_final_message()
    return _text(message)
