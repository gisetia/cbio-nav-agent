"""Utility functions to emit OpenAI-style chat completion payloads."""

from __future__ import annotations

import json
import time
import uuid
from typing import Iterable, List, Sequence


def _base_payload(model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
    }


def sse_chat_completions(
    chunk_iterable: Iterable[str], model: str
) -> Iterable[str]:
    """Yield SSE-formatted chunks that mimic the OpenAI Chat Completions API."""
    base = _base_payload(model)
    for chunk in chunk_iterable:
        payload = {
            **base,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(payload)}\n\n"

    yield "data: [DONE]\n\n"


def chat_completion_response(
    chunk_iterable: Sequence[str], model: str, prompt_messages: Sequence[str]
) -> dict:
    """Return a one-shot JSON response matching the OpenAI API shape."""
    content = "".join(chunk_iterable)
    payload = _base_payload(model)
    payload.update(
        {
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "prompt": prompt_messages,
                }
            ],
        }
    )
    return payload
