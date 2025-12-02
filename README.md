# cBio Navigator Agent

Claude-powered agent that can reach tools served by the MCP endpoint at `http://cbioportal-navigator:8002/mcp` and exposes an OpenAI-compatible `/chat/completions` API.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
export ANTHROPIC_API_KEY=your_key_here
uvicorn cbio_nav_agent.api:app --host 0.0.0.0 --port 5000 --reload
```

### Request format

Matches the reference `cbioportal_mcp_qa.api` module. POST to `/chat/completions` with:

```json
{
  "model": "anthropic:claude-3-5-sonnet-20240620",  // or "cbionav" alias
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true
}
```

### What it does

- Discovers tools from the MCP server `http://cbioportal-navigator:8002/mcp`.
- Runs a Claude tool-using loop to satisfy the user request with full conversation history.
- Streams responses using the OpenAI chat completions shape.

If you need to target a different MCP endpoint, set `MCP_SERVER_URL`.

### Configuration

- `ANTHROPIC_API_KEY` (required): API key for Claude.
- `MCP_SERVER_URL` (optional): Defaults to `http://cbioportal-navigator:8002/mcp`. In Docker on macOS/Windows, use `http://host.docker.internal:8002/mcp` to reach the host.
- `DEFAULT_MODEL` (optional): Override the default Claude model. The alias `cbionav` maps to this value.
- `DEFAULT_TEMPERATURE`, `DEFAULT_MAX_OUTPUT_TOKENS`, `DEFAULT_CHUNK_SIZE` (optional): Set fixed generation parameters (not user-overridable per request).

### Notes

- The API is intentionally minimal; extend the agent logic in `cbio_nav_agent/agent.py` as needed.
- The service defaults to streaming responses; set `"stream": false` to receive one JSON payload.
- The default system prompt lives in `prompts/system_prompt.txt` (loaded from the current working directory or `SYSTEM_PROMPT_FILE`). Edit that file to change the agent’s initial instructions.
- Override model via the API `model` field (or the `cbionav` alias) or `DEFAULT_MODEL` env var. Other generation parameters are fixed via settings/env and not user-controllable per request.

### Docker

```bash
docker compose up --build
```

- The container listens on port `5000` by default (adjust the compose `ports` mapping if you need a different host port).

- Ensure `ANTHROPIC_API_KEY` and `MCP_SERVER_URL` are set in `.env` or the compose `environment`. Use `host.docker.internal` to reach a host-running MCP server on macOS/Windows, or your host IP on Linux.

### LibreChat

You can connect LibreChat as a custom OpenAI-compatible endpoint:

```yaml
endpoints:
  custom:
    - name: "cBioNavAgent - API"
      apiKey: "none"
      baseURL: "http://<host>:5000"
      models:
        default: ["cBioNav"]
      titleConvo: true
      titleModel: "cBioNavAgent - API"
      modelDisplayLabel: "cBioNavAgent - API"
```

Replace `<host>` with the agent host (e.g., `localhost` when port-forwarded). The `cbionav` model alias maps to your default model.
