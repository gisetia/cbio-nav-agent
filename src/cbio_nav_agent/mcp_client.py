"""Thin MCP client used by the Claude agent to discover and call tools."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .settings import CLIENT_NAME


@dataclass
class MCPToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]

    def to_anthropic(self) -> Dict[str, Any]:
        """Return the format expected by the Anthropic Messages API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class MCPClient:
    """Wrapper around the MCP SSE client."""

    def __init__(self, server_url: str):
        self.server_url = server_url
        try:
            from mcp.client.session import ClientSession  # type: ignore
            from mcp.client.streamable_http import streamablehttp_client  # type: ignore
            from mcp import types  # type: ignore
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise RuntimeError(
                "The 'mcp' package is required to talk to MCP servers. "
                "Install it with `pip install mcp`."
            ) from exc

        self._ClientSession = ClientSession
        self._transport_client = streamablehttp_client
        self._types = types

    @asynccontextmanager
    async def _session(self):
        """Yield an initialized MCP session."""
        async with self._transport_client(self.server_url) as (
            read_stream,
            write_stream,
            _get_session_id,
        ):
            client_info = self._types.Implementation(name=CLIENT_NAME, version="0.1.0")
            async with self._ClientSession(
                read_stream, write_stream, client_info=client_info
            ) as session:
                await session.initialize()
                yield session

    async def list_tools(self) -> List[MCPToolDefinition]:
        """Return the tool definitions exposed by the MCP server."""
        async with self._session() as session:
            response = await session.list_tools()
            tools = []
            for tool in response.tools:
                # The MCP types use camelCase; Anthropic expects snake_case.
                input_schema = getattr(tool, "inputSchema", None) or getattr(
                    tool, "input_schema", None
                )
                if input_schema is None:
                    input_schema = {"type": "object"}
                tools.append(
                    MCPToolDefinition(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=input_schema,
                    )
                )
            return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Invoke a specific tool and return a readable string result."""
        async with self._session() as session:
            response = await session.call_tool(tool_name, arguments)
            # Collect text content; fall back to a repr if necessary.
            rendered: List[str] = []
            for item in response.content:
                text = getattr(item, "text", None) or getattr(item, "data", None)
                if text is None:
                    text = str(item)
                rendered.append(text)
            return "\n".join(rendered)


async def discover_tools(server_url: str) -> List[MCPToolDefinition]:
    """Convenience helper to fetch tools outside of a class instance."""
    client = MCPClient(server_url)
    return await client.list_tools()


async def call_tool(server_url: str, tool_name: str, arguments: Dict[str, Any]) -> str:
    """Convenience helper to invoke a tool outside of a class instance."""
    client = MCPClient(server_url)
    return await client.call_tool(tool_name, arguments)
