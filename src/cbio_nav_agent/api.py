"""FastAPI wrapper exposing a Claude+MCP agent with an OpenAI-compatible interface."""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, StreamingResponse

from .agent import ClaudeMCPAgent
from .openai_adapter import (
    async_sse_chat_completions,
    chat_completion_response,
    sse_chat_completions,
)
from .settings import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MCP_SERVER_URL,
    DEFAULT_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    ENABLE_STEP_STREAMING,
)

app = FastAPI()

MODEL_ALIASES = {
    # Maps friendly names to provider-qualified models.
    "cbionav": DEFAULT_MODEL,
}

class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., user).")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[Message]
    stream: bool = True
    api_key: Optional[str] = None
    mcp_server_url: str = DEFAULT_MCP_SERVER_URL
    system_prompt: Optional[str] = None


def _resolve_model(model: str) -> str:
    """Resolve incoming model aliases to provider-qualified model names."""
    if not model:
        return DEFAULT_MODEL
    normalized = model.strip().lower()
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]
    if ":" not in model:
        return DEFAULT_MODEL
    return model


def _extract_user_question(messages: List[Message]) -> Optional[str]:
    """Return the most recent user message content."""
    for message in reversed(messages):
        if message.role.lower() == "user":
            return message.content
    return None


def _chunk_answer(answer: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """Split the answer into smaller pieces to feed the SSE adapter."""
    if not answer:
        return [""]
    return [answer[i : i + chunk_size] for i in range(0, len(answer), chunk_size)]


@app.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    print("-- -- [api] incoming request", req.model_dump(exclude_none=True))

    resolved_model = _resolve_model(req.model)
    resolved_system_prompt = (
        req.system_prompt if req.system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    )

    question = _extract_user_question(req.messages)
    if not question:
        raise HTTPException(
            status_code=400, detail="A user message is required to ask a question."
        )

    # Convert incoming Pydantic messages to the format expected by the Anthropic client.
    conversation = [
        {"role": message.role, "content": message.content} for message in req.messages
    ]

    try:
        agent = ClaudeMCPAgent(
            api_key=req.api_key,
            model=resolved_model,
            mcp_server_url=req.mcp_server_url,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            system_prompt=resolved_system_prompt,
            temperature=DEFAULT_TEMPERATURE,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if req.stream:
        print(
            "-- -- [api] streaming response body",
            {"model_requested": req.model, "model_used": resolved_model, "step_streaming": ENABLE_STEP_STREAMING},
        )
        if ENABLE_STEP_STREAMING:
            stream = agent.ask_stream(conversation)
            return StreamingResponse(
                async_sse_chat_completions(stream, model=agent.model),
                media_type="text/event-stream",
            )
        # Fallback: stream only the final answer in OpenAI-style chunks.
        answer = await agent.ask(conversation)
        chunks = _chunk_answer(answer, chunk_size=DEFAULT_CHUNK_SIZE)
        return StreamingResponse(
            sse_chat_completions(chunks, model=agent.model),
            media_type="text/event-stream",
        )

    answer = await agent.ask(conversation)
    chunks = _chunk_answer(answer, chunk_size=DEFAULT_CHUNK_SIZE)
    payload = chat_completion_response(
        chunk_iterable=chunks,
        model=agent.model,
        prompt_messages=[message.content for message in req.messages],
    )
    print(
        "-- -- [api] json response body",
        {"model_requested": req.model, "model_used": resolved_model, "payload": payload},
    )
    return JSONResponse(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("cbio_nav_agent.api:app", host="0.0.0.0", port=4000, reload=False)
