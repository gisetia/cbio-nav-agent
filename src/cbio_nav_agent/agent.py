"""Claude agent that loops through MCP tool calls until a final answer is ready."""

from __future__ import annotations

import logging
import os
import json
from typing import AsyncIterable, List, Optional

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolResultBlockParam, ToolUseBlock

from .mcp_client import MCPClient
from .settings import (
    DEFAULT_MCP_SERVER_URL,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    ENABLE_FORMATTING,
    STREAM_TEXT_LIVE,
    STREAM_TOOL_NOTICES_LIVE,
    STREAM_TOOL_ARGS_LIVE,
    STREAM_TOOL_RESPONSES_LIVE,
    INCLUDE_FINAL_TEXT,
    INCLUDE_TOOL_LOGS_FINAL,
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
        simulate: bool = False,
    ):
        self.simulate = simulate
        self.model = self._normalize_model(model)
        self.max_output_tokens = max_output_tokens
        self.system_prompt = system_prompt or ""
        self.temperature = temperature
        logger.info(
            "Using system prompt",
            extra={"system_prompt_preview": (self.system_prompt[:120] if self.system_prompt else "<empty>")},
        )

        if self.simulate:
            self.api_key = None
            self.client = None
            self.mcp_client = None
            logger.info("Simulation mode enabled; responding with fake data.")
            return

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required.")

        self.client = AsyncAnthropic(api_key=self.api_key)
        self.mcp_client = MCPClient(mcp_server_url)

    @staticmethod
    def _format_system_content(system_prompt: Optional[str]):
        """Return system content in the block format Anthropic expects."""
        if not system_prompt:
            return None
        return [{"type": "text", "text": str(system_prompt)}]

    async def ask_stream(
        self,
        messages: List[MessageParam],
        *,
        stream_mode: bool = True,
        include_tool_logs_final: bool = INCLUDE_TOOL_LOGS_FINAL,
        format_enabled: Optional[bool] = None,
        stream_text_live: Optional[bool] = None,
        stream_tool_notices_live: Optional[bool] = None,
        stream_tool_args_live: Optional[bool] = None,
        stream_tool_responses_live: Optional[bool] = None,
    ) -> AsyncIterable[str]:
        """Yield intermediate and final responses while invoking MCP tools.

        stream_mode=True streams live pieces; stream_mode=False emits only final pieces in order.
        """
        if self.simulate:
            eff_format = ENABLE_FORMATTING if format_enabled is None else format_enabled
            fake_text = self._build_simulated_response(messages)
            yield self._format_final_text(fake_text, eff_format)
            return

        logger.info(
            "Starting ask",
            extra={"model": self.model, "mcp_server": self.mcp_client.server_url},
        )
        tools = [tool.to_anthropic() for tool in await self.mcp_client.list_tools()]
        logger.info("Discovered %d MCP tools", len(tools))

        # Work on a shallow copy so we don't mutate the caller's list.
        conversation: List[MessageParam] = list(messages)
        answer_parts: List[str] = []
        tool_logs: List[str] = []
        ordered_events: List[str] = []  # used when stream_mode is False to preserve order

        # Resolve effective streaming flags (allow per-call overrides).
        eff_stream_text_live = STREAM_TEXT_LIVE if stream_text_live is None else stream_text_live
        eff_stream_tool_notices_live = (
            STREAM_TOOL_NOTICES_LIVE if stream_tool_notices_live is None else stream_tool_notices_live
        )
        eff_stream_tool_args_live = (
            STREAM_TOOL_ARGS_LIVE if stream_tool_args_live is None else stream_tool_args_live
        )
        eff_stream_tool_responses_live = (
            STREAM_TOOL_RESPONSES_LIVE if stream_tool_responses_live is None else stream_tool_responses_live
        )
        eff_format = ENABLE_FORMATTING if format_enabled is None else format_enabled

        while True:
            logger.info(
                "Claude turn",
                extra={
                    "messages": conversation,
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
            )
            system_content = self._format_system_content(self.system_prompt)
            request_kwargs = dict(
                model=self.model,
                max_tokens=self.max_output_tokens,
                tools=tools,
                temperature=self.temperature,
                messages=conversation,
            )
            if system_content is not None:
                request_kwargs["system"] = system_content

            response = await self.client.messages.create(**request_kwargs)

            conversation.append({"role": "assistant", "content": response.content})

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
                is_intermediate = bool(tool_requests)
                formatted_text = (
                    self._format_intermediate_text(combined_text, eff_format)
                    if is_intermediate
                    else self._format_final_text(combined_text, eff_format)
                )
                if stream_mode and eff_stream_text_live:
                    yield formatted_text
                elif not stream_mode and INCLUDE_FINAL_TEXT:
                    ordered_events.append(formatted_text)

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
                if stream_mode and eff_stream_tool_notices_live:
                    yield self._format_tool_notice(tool_use.name, eff_format)
                elif not stream_mode:
                    ordered_events.append(self._format_tool_notice(tool_use.name, eff_format))
                tool_output = await self.mcp_client.call_tool(
                    tool_name=tool_use.name, arguments=tool_use.input or {}
                )
                logger.info(
                    "Result from %s:\n%s",
                    tool_use.name,
                    (tool_output or "").strip(),
                )
                if stream_mode:
                    if eff_stream_tool_args_live:
                        yield self._format_tool_args(tool_use.name, tool_use.input or {}, eff_format)
                    if eff_stream_tool_responses_live:
                        yield self._format_tool_response(tool_use.name, tool_output, eff_format)
                    if include_tool_logs_final:
                        if not eff_stream_tool_args_live:
                            tool_logs.append(self._format_tool_args(tool_use.name, tool_use.input or {}, eff_format))
                        if not eff_stream_tool_responses_live:
                            tool_logs.append(self._format_tool_response(tool_use.name, tool_output, eff_format))
                else:
                    if include_tool_logs_final:
                        ordered_events.append(self._format_tool_args(tool_use.name, tool_use.input or {}, eff_format))
                        ordered_events.append(self._format_tool_response(tool_use.name, tool_output, eff_format))
                tool_results.append(
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=tool_use.id,
                        content=tool_output or "No content returned by tool.",
                    )
                )

            # Feed tool results back to Claude for another reasoning step.
            conversation.append({"role": "user", "content": tool_results})

        # Append collected pieces.
        if stream_mode:
            if include_tool_logs_final:
                for log_entry in tool_logs:
                    yield log_entry
            final_answer = "".join(answer_parts).strip()
            if final_answer and INCLUDE_FINAL_TEXT and not STREAM_TEXT_LIVE:
                yield f"\n\n{final_answer}"
        else:
            for evt in ordered_events:
                yield evt

    async def ask(
        self,
        messages: List[MessageParam],
        *,
        include_tool_logs_final: bool = INCLUDE_TOOL_LOGS_FINAL,
        format_enabled: Optional[bool] = None,
        stream_text_live: Optional[bool] = None,
        stream_tool_notices_live: Optional[bool] = None,
        stream_tool_args_live: Optional[bool] = None,
        stream_tool_responses_live: Optional[bool] = None,
    ) -> str:
        """Collect all streamed chunks into a single answer string."""
        parts: List[str] = []
        async for chunk in self.ask_stream(
            messages,
            stream_mode=False,
            include_tool_logs_final=include_tool_logs_final,
            format_enabled=format_enabled,
            stream_text_live=stream_text_live,
            stream_tool_notices_live=stream_tool_notices_live,
            stream_tool_args_live=stream_tool_args_live,
            stream_tool_responses_live=stream_tool_responses_live,
        ):
            if parts:
                parts.append("\n\n")
            parts.append(chunk)
        return "".join(parts).strip()

    @staticmethod
    def _normalize_model(model: str) -> str:
        """Drop provider prefixes like 'anthropic:' so the client accepts the model name."""
        normalized = (model or DEFAULT_MODEL).strip()
        if normalized.startswith("anthropic:"):
            return normalized.split(":", 1)[1]
        return normalized

    @staticmethod
    def _format_intermediate_text(text: str, enable: bool) -> str:
        """Customize formatting for in-between (assistant) text before a tool call."""
        return text if not enable else f"\n\n*{text}*"

    @staticmethod
    def _format_final_text(text: str, enable: bool) -> str:
        """Customize formatting for final assistant text (no more tool calls)."""
        return text if not enable else f"\n\n{text}"

    @staticmethod
    def _format_tool_notice(tool_name: str, enable: bool) -> str:
        if not enable:
            return f"Calling {tool_name}"
        return f"\n\n*Calling* `{tool_name}` *tool*"

    @staticmethod
    def _format_tool_args(tool_name: str, args: dict, enable: bool) -> str:
        payload = json.dumps(args, separators=(",", ":"))
        if not enable:
            return f"Args for {tool_name}: {payload}"
        return f" *with args:*\n```json\n{payload}\n```"

    @staticmethod
    def _format_tool_response(tool_name: str, output: Optional[str], enable: bool) -> str:
        raw_content = (output or "No content returned by tool.").strip()
        compact_content = ClaudeMCPAgent._compact_json(raw_content)
        content = compact_content or raw_content
        if not enable:
            return f"Response from {tool_name}: {content}"
        return f"\n\n*Response from* `{tool_name}` *tool:*\n```json\n{content}\n```"

    @staticmethod
    def _build_simulated_response(messages: List[MessageParam]) -> str:
        user_message = ""
        for message in reversed(messages):
            if message.get("role", "").lower() == "user":
                user_message = str(message.get("content", "") or "")
                break
        if not user_message:
            user_message = "<no user message provided>"
        return (
            "[SIMULATION MODE] Returning a canned response. No Anthropic or MCP calls were made.\n\n"
            "## Quick preview\n"
            f"- **User said:** {user_message}\n"
            "- _Italic line for styling check._\n"
            "- Inline code: `foo.bar(baz)`\n\n"
            "### Code block\n"
            "```python\n"
            "def example():\n"
            "    return \"sample output\"\n"
            "```\n\n"
            "### Table\n"
            "| Column | Value |\n"
            "| --- | --- |\n"
            "| A | alpha |\n"
            "| B | beta |\n\n"
            "> Blockquote to test nested look.\n\n"
            "1. Ordered item one\n"
            "2. Ordered item two\n"
            "3. Ordered item three\n\n"
            "<details>\n"
            "<summary>HTML details summary (common GitHub-style collapse)</summary>\n\n"
            "Hidden text inside a collapsible block. Useful for long notes or logs.\n\n"
            "```\n"
            "Multi-line content can live here.\n"
            "```\n"
            "</details>\n\n"
            "::: details Markdown-it style collapsible (if supported)\n"
            "Hidden text using the `details` admonition.\n"
            "### Nested heading\n"
            "- bullet one\n"
            "- bullet two\n"
            "```\n"
            "code stays hidden too\n"
            "```\n"
            "::: \n\n"
            "!!! note \"Admonition with title (mkdocs/markdown-extras)\"\n"
            "    This may render as a collapsible panel in some renderers.\n"
            "    Indented markdown stays inside.\n"
            "    - item A\n"
            "    - item B\n\n"
            "???+ info \"MkDocs-style toggle (foldable if supported)\"\n"
            "    This uses the `???` syntax common in MkDocs Material.\n"
            "    It only works if the renderer enables that extension.\n\n"
            "[details=GitLab-style summary]\n"
            "GitLab-flavored Markdown supports this details block.\n"
            "More lines here.\n"
            "[/details]\n\n"
            ">! Spoiler syntax (Discord/Reddit style). Replace this with your hidden text. !<\n\n"
            "```\n"
            "<!-- Plain HTML comment can be ignored by most renderers; not collapsible but hidden in view -->\n"
            "```\n\n"
            "::: spoiler Click to view (if `spoiler` fence is supported)\n"
            "Hidden spoiler content.\n"
            "::: \n\n"
            "[spoiler]\n"
            "BBCode-style spoiler block. Some renderers collapse this by default.\n"
            "[/spoiler]\n\n"
            "<spoiler>HTML spoiler tag; may blur or hide text if supported.</spoiler>\n\n"
            "- [ ] Checkbox trick: some UIs let you fold long lists behind a checkbox label.\n"
            "  Hidden-ish content after a checkbox label.\n\n"
            "> **Pseudo-collapse:** Some UIs only allow blockquotes; you can simulate collapse by\n"
            "> writing a short summary and keeping the rest below it.\n\n"
            "---\n"
            "_End of simulated content._"
        )

    @staticmethod
    def _compact_json(payload: str) -> Optional[str]:
        """Return a minified JSON string if the payload parses as JSON."""
        try:
            parsed = json.loads(payload)
        except Exception:
            return None
        try:
            return json.dumps(parsed, separators=(",", ":"))
        except Exception:
            return None
