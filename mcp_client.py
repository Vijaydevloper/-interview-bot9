"""Minimal synchronous MCP client: one stdio round trip per tool call."""
import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
SERVERS = {
    "progress": "progress_server.py",
    "docs": "docs_server.py",
    "notes": "notes_server.py",
    "rag": "rag_server.py",
}


async def _call(server: str, tool: str, args: dict):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(HERE / SERVERS[server])],
        env={**os.environ},
        cwd=str(Path.cwd()),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(tool, args)
            text = res.content[0].text if res.content else "null"
            if getattr(res, "is_error", getattr(res, "isError", False)):
                raise RuntimeError(text)
            return json.loads(text)


def call(server: str, tool: str, **args):
    return asyncio.run(_call(server, tool, args))
