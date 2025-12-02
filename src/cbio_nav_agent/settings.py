"""Configuration defaults for the cBio Navigator agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a local .env file if present.
load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic:claude-sonnet-4-20250514")
DEFAULT_MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL", "http://cbioportal-navigator:8002/mcp"
)
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1200"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("DEFAULT_MAX_OUTPUT_TOKENS", "1024"))
# Toggle step-by-step streaming of thoughts/tool calls (True by default).
ENABLE_STEP_STREAMING = os.getenv("ENABLE_STEP_STREAMING", "true").lower() in {
    "1",
    "true",
    "yes",
}
# Resolve a default prompt path, preferring an explicit env var.
ENV_SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE")


def load_system_prompt() -> str:
    """Load the system prompt from a file if present."""
    candidate_paths = []

    if ENV_SYSTEM_PROMPT_FILE:
        candidate_paths.append(Path(ENV_SYSTEM_PROMPT_FILE))

    # Common locations: working directory and project root when installed.
    candidate_paths.append(Path.cwd() / "prompts" / "system_prompt.txt")
    candidate_paths.append(Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt")

    for prompt_path in candidate_paths:
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()

    return ""


DEFAULT_SYSTEM_PROMPT = load_system_prompt()

# Keep the user-agent identifier simple so MCP servers can see who is calling.
CLIENT_NAME = os.getenv("CLIENT_NAME", "cbio-nav-agent")
