"""RAG MCP server (stdio). Run: python rag_server.py"""
from mcp_compat import make_server

import rag_tools as r

mcp = make_server("rag")
for fn in (r.search_notes, r.sample_notes, r.list_topics, r.library_summary):
    mcp.tool()(fn)

if __name__ == "__main__":
    mcp.run()
