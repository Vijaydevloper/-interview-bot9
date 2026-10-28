"""Progress MCP server (stdio). Run: python progress_server.py"""
from mcp_compat import make_server

import progress_tools as p

mcp = make_server("progress")
for fn in (p.log_answer, p.log_answers, p.get_weak_topics, p.get_topic_stats, p.get_history):
    mcp.tool()(fn)

if __name__ == "__main__":
    mcp.run()
