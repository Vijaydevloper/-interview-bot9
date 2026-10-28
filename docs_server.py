"""Web/Docs MCP server (stdio). Run: python docs_server.py"""
from mcp_compat import make_server

import docs_tools as d

mcp = make_server("docs")
for fn in (d.search_docs, d.fetch_doc):
    mcp.tool()(fn)

if __name__ == "__main__":
    mcp.run()
