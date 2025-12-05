# cBio Navigator Agent

Agent that can reach tools served by the MCP endpoint at `http://cbioportal-navigator:8002/mcp` and exposes an OpenAI-compatible `/chat/completions` API. Anthropic models currently support the MCP tool-use loop.

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
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true,
  "full_stream": false
}
```

Example `curl` call (streaming):

```bash
curl -X POST http://localhost:5000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "stream": true,
    "full_stream": true,
    "formatting": false,
    "messages": [
      {"role": "user", "content": "What is the median survival time in the Pediatric Neuroblastoma study from TARGET?"}
    ]
  }'

```

### What it does

- Discovers tools from the MCP server `http://cbioportal-navigator:8002/mcp`.
- Runs a Claude tool-using loop to satisfy the user request with full conversation history.
- Streams responses using the OpenAI chat completions shape.

### Configuration

- `ANTHROPIC_API_KEY` (required for Anthropic models with tool use).
- `MCP_SERVER_URL` (optional): Defaults to `http://cbioportal-navigator:8002/mcp`. In Docker on macOS/Windows, use `http://host.docker.internal:8002/mcp` to reach the host.
- `DEFAULT_MODEL` (optional): Override the default model (applied server-side; requests cannot override). Tool-use loop currently works with Anthropic models.
- `DEFAULT_TEMPERATURE`, `DEFAULT_MAX_OUTPUT_TOKENS`, `DEFAULT_CHUNK_SIZE` (optional): Set fixed generation parameters (not user-overridable per request).
- Streaming controls (env): `ENABLE_STEP_STREAMING`, `STREAM_TEXT_LIVE`, `STREAM_TOOL_NOTICES_LIVE`, `STREAM_TOOL_ARGS_LIVE`, `STREAM_TOOL_RESPONSES_LIVE`, `INCLUDE_FINAL_TEXT`, `INCLUDE_TOOL_LOGS_FINAL`.
- Formatting: `ENABLE_FORMATTING` (env, default true); per-request override via `"formatting": false|true`.

### Notes

- The API is intentionally minimal; extend the agent logic in `cbio_nav_agent/agent.py` as needed.
- The service defaults to streaming responses; set `"stream": false` to receive one JSON payload.
- The default system prompt lives in `prompts/system_prompt.txt` (loaded from the current working directory or `SYSTEM_PROMPT_FILE`). Edit that file to change the agent’s initial instructions.
- Requests accept `messages`, `stream`, and optional `"full_stream"` and `"formatting"`; model/API key/MCP URL/system prompt are set server-side via environment.

### Docker

```bash
docker compose up --build
```

- The container listens on port `5000` by default (adjust the compose `ports` mapping if you need a different host port).

- Ensure `ANTHROPIC_API_KEY` and `MCP_SERVER_URL` are set in `.env` or the compose `environment`. Use `host.docker.internal` to reach a host-running MCP server on macOS/Windows, or your host IP on Linux.

### Docker usage

- Build and run: `docker compose up --build`
- Environment to set (e.g., via `.env`):
  - `ANTHROPIC_API_KEY` (required)
  - `MCP_SERVER_URL` (defaults to `http://host.docker.internal:8002/mcp` in compose)
  - Streaming/formatting env flags: `ENABLE_STEP_STREAMING`, `STREAM_TEXT_LIVE`, `STREAM_TOOL_NOTICES_LIVE`, `STREAM_TOOL_ARGS_LIVE`, `STREAM_TOOL_RESPONSES_LIVE`, `INCLUDE_FINAL_TEXT`, `INCLUDE_TOOL_LOGS_FINAL`, `ENABLE_FORMATTING`
- Per-request overrides: set `"full_stream": true` and/or `"formatting": true|false` in the request payload
- API available at `http://localhost:5000/chat/completions` unless you remap the port

### Streaming modes

- By default, streaming behavior follows server env settings.
- Set `"full_stream": true` in a request to force live streaming of assistant text and tool notices/args/responses and include tool logs in the output (overrides env).
- To disable step streaming globally and only stream the final answer in OpenAI-style chunks, set `ENABLE_STEP_STREAMING=false` in the environment.

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
