"""Shared OpenAI-compatible chat helper.

Why this exists: three call sites (MBG extraction, example agent, LLM verifier) each hand-rolled
the same request, and all of them broke the same way on thinking servers (e.g. a local
llama.cpp/strata Qwen "Flash-Next" build):

* Omitting `max_tokens` lets the server generate to the full context window (262k here), so one
  call burned >20 minutes and still returned `content: null` because it never finished reasoning;
* thinking models return the final answer in `message.content` only after reasoning, and expose
  the chain in `message.reasoning_content`. A `None` content must not become an AttributeError.

`chat_text` fixes both: explicit token budget, optional reasoning effort, content/reasoning
fallback, and an error that says what actually happened.
"""

from __future__ import annotations

import json
from typing import Any

import requests

DEFAULT_MAX_TOKENS = 4096


def chat_text(
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str | None = None,
    timeout: int = 240,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """POST /chat/completions and return (text, meta).

    `text` is message.content, falling back to reasoning_content when a thinking server leaves
    content null (the JSON we want may already be in the reasoning text).
    `meta` = {"usage": ..., "finish_reason": ...} — finish_reason == "length" means the answer was
    cut off, which is the difference between 'the model is bad at JSON' and 'we ran out of budget'.
    """
    payload: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": int(max_tokens)}
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in chat response: {str(data)[:300]}")
    message = choices[0].get("message") or {}
    finish = choices[0].get("finish_reason")
    usage = data.get("usage") or {}

    text = message.get("content")
    if not text:
        text = message.get("reasoning_content")
        if text:
            text = str(text)
    if not text:
        raise RuntimeError(
            "Empty chat completion (content and reasoning_content both null). "
            f"finish_reason={finish!r} usage={usage} max_tokens={payload['max_tokens']}. "
            "On thinking servers a too-small max_tokens truncates the reasoning phase and "
            "leaves content null; raise max_tokens or set a reasoning effort."
        )
    return str(text), {"usage": usage, "finish_reason": finish}
