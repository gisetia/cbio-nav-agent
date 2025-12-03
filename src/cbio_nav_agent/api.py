"""FastAPI wrapper exposing a Claude+MCP agent with an OpenAI-compatible interface."""

from __future__ import annotations

from typing import List, Optional
import os

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
    INCLUDE_TOOL_LOGS_FINAL,
)

app = FastAPI()

class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., user).")
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: bool = True
    full_stream: Optional[bool] = None


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

    # Requests cannot override core settings; use defaults.
    resolved_model = DEFAULT_MODEL
    resolved_system_prompt = DEFAULT_SYSTEM_PROMPT
    step_streaming = ENABLE_STEP_STREAMING
    include_tool_logs_final = INCLUDE_TOOL_LOGS_FINAL
    # If full_stream is requested, stream everything live and include tool logs.
    if req.full_stream:
        step_streaming = True
        include_tool_logs_final = True
        stream_text_live = True
        stream_tool_notices_live = True
        stream_tool_args_live = True
        stream_tool_responses_live = True
    else:
        stream_text_live = None
        stream_tool_notices_live = None
        stream_tool_args_live = None
        stream_tool_responses_live = None

    question = _extract_user_question(req.messages)
    if not question:
        raise HTTPException(
            status_code=400, detail="A user message is required to ask a question."
        )

    # Convert incoming Pydantic messages to the format expected by the Anthropic client.
    conversation = [
        {"role": message.role, "content": message.content} for message in req.messages
    ]

    agent = ClaudeMCPAgent(
        api_key=None,
        model=resolved_model,
        mcp_server_url=DEFAULT_MCP_SERVER_URL,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        system_prompt=resolved_system_prompt,
        temperature=DEFAULT_TEMPERATURE,
    )

    if req.stream:
        print(
            "-- -- [api] streaming response body",
            {"model_used": resolved_model, "step_streaming": step_streaming},
        )
        if step_streaming:
            stream = agent.ask_stream(
                conversation,
                include_tool_logs_final=include_tool_logs_final,
                stream_text_live=stream_text_live,
                stream_tool_notices_live=stream_tool_notices_live,
                stream_tool_args_live=stream_tool_args_live,
                stream_tool_responses_live=stream_tool_responses_live,
            )
            return StreamingResponse(
                async_sse_chat_completions(stream, model=agent.model),
                media_type="text/event-stream",
            )
        # Fallback: stream only the final answer in OpenAI-style chunks.
        answer = await agent.ask(
            conversation,
            include_tool_logs_final=include_tool_logs_final,
            stream_text_live=stream_text_live,
            stream_tool_notices_live=stream_tool_notices_live,
            stream_tool_args_live=stream_tool_args_live,
            stream_tool_responses_live=stream_tool_responses_live,
        )
        chunks = _chunk_answer(answer, chunk_size=DEFAULT_CHUNK_SIZE)
        return StreamingResponse(
            sse_chat_completions(chunks, model=agent.model),
            media_type="text/event-stream",
        )

    answer = await agent.ask(
        conversation,
        include_tool_logs_final=include_tool_logs_final,
        stream_text_live=stream_text_live,
        stream_tool_notices_live=stream_tool_notices_live,
        stream_tool_args_live=stream_tool_args_live,
        stream_tool_responses_live=stream_tool_responses_live,
    )
    chunks = _chunk_answer(answer, chunk_size=DEFAULT_CHUNK_SIZE)
    payload = chat_completion_response(
        chunk_iterable=chunks,
        model=agent.model,
        prompt_messages=[message.content for message in req.messages],
    )
    print(
        "-- -- [api] json response body",
        {"model_used": resolved_model, "payload": payload},
    )
    return JSONResponse(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("cbio_nav_agent.api:app", host="0.0.0.0", port=5000, reload=False)
