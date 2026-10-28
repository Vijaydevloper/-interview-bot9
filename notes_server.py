"""Notes MCP server (stdio, optional). Run: python notes_server.py"""
from mcp_compat import make_server

import notes_tools as n

mcp = make_server("notes")
mcp.tool()(n.save_summary)

if __name__ == "__main__":
    mcp.run()
