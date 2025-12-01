# cBio Navigator Agent

Claude-powered agent that can reach tools served by the MCP endpoint at `http://cbioportal-navigator:8002/mcp` and exposes an OpenAI-compatible `/chat/completions` API.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
export ANTHROPIC_API_KEY=your_key_here
uvicorn cbio_nav_agent.api:app --host 0.0.0.0 --port 4000 --reload
```

### Request format

Matches the reference `cbioportal_mcp_qa.api` module. POST to `/chat/completions` with:

```json
{
  "model": "anthropic:claude-3-5-sonnet-20240620",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true
}
```

### What it does

- Discovers tools from the MCP server `http://cbioportal-navigator:8002/mcp`.
- Runs a Claude tool-using loop to satisfy the user request.
- Streams responses using the OpenAI chat completions shape.

If you need to target a different MCP endpoint, set `MCP_SERVER_URL`.

### Configuration

- `ANTHROPIC_API_KEY` (required): API key for Claude.
- `MCP_SERVER_URL` (optional): Defaults to `http://cbioportal-navigator:8002/mcp`.
- `DEFAULT_MODEL` (optional): Override the default Claude model.

### Notes

- The API is intentionally minimal; extend the agent logic in `cbio_nav_agent/agent.py` as needed.
- The service defaults to streaming responses; set `"stream": false` to receive one JSON payload.
- The default system prompt lives in `prompts/system_prompt.txt`. Edit that file (or point `SYSTEM_PROMPT_FILE` to another tracked file) to change the agent’s initial instructions.
- Override model via the API `model` field or `DEFAULT_MODEL` env var. Other generation parameters (temperature, max output tokens, chunk size) are fixed via settings/env and not user-controllable per request.
