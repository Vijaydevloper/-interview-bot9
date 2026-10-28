"""Tool access layer used by the app.

Calls go through MCP (stdio) by default. Set USE_MCP=0, or run without the
`mcp` package installed, to call the same functions in-process.
"""
import json
import os

import docs_tools
import notes_tools
import progress_tools
import rag_tools

_DIRECT = {"progress": progress_tools, "docs": docs_tools, "notes": notes_tools, "rag": rag_tools}


def call(server: str, tool: str, **args):
    if os.environ.get("USE_MCP", "1") == "1":
        try:
            import mcp_client

            return mcp_client.call(server, tool, **args)
        except ImportError:
            pass
    return json.loads(getattr(_DIRECT[server], tool)(**args))
