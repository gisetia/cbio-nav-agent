"""Claude agent that loops through MCP tool calls until a final answer is ready."""

from __future__ import annotations

import logging
import os
import json
from typing import List, Optional

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolResultBlockParam, ToolUseBlock

from .mcp_client import MCPClient
from .settings import (
    DEFAULT_MCP_SERVER_URL,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
)

# Simple module logger; defaults to INFO if not configured by the host app.
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
# Quiet noisy third-party loggers; keep our own logs at INFO.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)
logging.getLogger("mcp.client").setLevel(logging.WARNING)


class ClaudeMCPAgent:
    """Runs a Claude tool-use loop backed by the MCP server."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        mcp_server_url: str = DEFAULT_MCP_SERVER_URL,
        max_output_tokens: int = 1024,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.model = self._normalize_model(model)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required.")

        self.client = AsyncAnthropic(api_key=self.api_key)
        self.mcp_client = MCPClient(mcp_server_url)
        self.max_output_tokens = max_output_tokens
        self.system_prompt = system_prompt or ""
        self.temperature = temperature

    @staticmethod
    def _format_system_content(system_prompt: Optional[str]):
        """Return system content in the block format Anthropic expects."""
        if not system_prompt:
            return None
        return [{"type": "text", "text": str(system_prompt)}]

    async def ask(self, question: str) -> str:
        """Ask Claude the question, invoking MCP tools when requested."""
        logger.info(
            "Starting ask",
            extra={"model": self.model, "mcp_server": self.mcp_client.server_url},
        )
        tools = [tool.to_anthropic() for tool in await self.mcp_client.list_tools()]
        logger.info("Discovered %d MCP tools", len(tools))

        messages: List[MessageParam] = [{"role": "user", "content": question}]
        answer_parts: List[str] = []

        while True:
            logger.info(
                "Claude turn",
                extra={"messages": messages, "temperature": self.temperature, "max_output_tokens": self.max_output_tokens},
            )
            system_content = self._format_system_content(self.system_prompt)
            request_kwargs = dict(
                model=self.model,
                max_tokens=self.max_output_tokens,
                tools=tools,
                temperature=self.temperature,
                messages=messages,
            )
            if system_content is not None:
                request_kwargs["system"] = system_content

            response = await self.client.messages.create(**request_kwargs)

            messages.append({"role": "assistant", "content": response.content})

            tool_requests = [
                block for block in response.content if isinstance(block, ToolUseBlock)
            ]
            text_blocks = [
                block.text for block in response.content if getattr(block, "text", None)
            ]

            # Preserve paragraph breaks between separate assistant text blocks.
            for text in text_blocks:
                if answer_parts:
                    answer_parts.append("\n\n")
                answer_parts.append(text)

            if text_blocks:
                combined_text = "\n\n".join(text_blocks).strip()
                # Friendly one-liner plus full text in structured extra for debugging.
                logger.info("Assistant reply:\n%s", combined_text)
                logger.debug(
                    "Assistant text detail",
                    extra={
                        "text": text_blocks,
                        "text_preview": combined_text[:200],
                        "text_length": len(combined_text),
                    },
                )

            if not tool_requests:
                logger.info("No tool requests, finishing.")
                break

            tool_results: List[ToolResultBlockParam] = []
            for tool_use in tool_requests:
                logger.info(
                    "Calling tool %s with args: %s",
                    tool_use.name,
                    json.dumps(tool_use.input or {}, indent=2),
                )
                tool_output = await self.mcp_client.call_tool(
                    tool_name=tool_use.name, arguments=tool_use.input or {}
                )
                logger.info(
                    "Result from %s:\n%s",
                    tool_use.name,
                    (tool_output or "").strip(),
                )
                tool_results.append(
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=tool_use.id,
                        content=tool_output or "No content returned by tool.",
                    )
                )

            # Feed tool results back to Claude for another reasoning step.
            messages.append({"role": "user", "content": tool_results})

        return "".join(answer_parts).strip()

    @staticmethod
    def _normalize_model(model: str) -> str:
        """Drop provider prefixes like 'anthropic:' so the client accepts the model name."""
        normalized = (model or DEFAULT_MODEL).strip()
        if normalized.startswith("anthropic:"):
            return normalized.split(":", 1)[1]
        return normalized
