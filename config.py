"""Loads settings from a `.env` file next to this file (e.g. GROQ_API_KEY).

Real environment variables win over .env, so `export GROQ_API_KEY=...` still works.
Every module that needs the key imports this first; MCP server processes inherit the result.
"""
import os
from pathlib import Path

ENV_FILE = Path(os.environ.get("ENV_FILE", Path(__file__).resolve().parent / ".env"))


def load_env(path: Path = ENV_FILE) -> bool:
    """Load KEY=VALUE lines from `path` without overriding variables that are already set."""
    if not path.is_file():
        return False
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return True
    except ImportError:  # python-dotenv not installed: minimal parser
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip("'\""))
        return True


load_env()
