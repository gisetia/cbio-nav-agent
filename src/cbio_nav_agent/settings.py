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
# Resolve the prompts directory relative to the repo root so uvicorn can be started anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT_FILE = Path(
    os.getenv(
        "SYSTEM_PROMPT_FILE",
        _PROJECT_ROOT / "prompts" / "system_prompt.txt",
    )
)


def load_system_prompt() -> str:
    """Load the system prompt from the prompt file."""
    prompt_path = Path(SYSTEM_PROMPT_FILE)
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()

    return ""


DEFAULT_SYSTEM_PROMPT = load_system_prompt()

# Keep the user-agent identifier simple so MCP servers can see who is calling.
CLIENT_NAME = os.getenv("CLIENT_NAME", "cbio-nav-agent")
