"""Create an MCP server object on both mcp 1.x (FastMCP) and mcp 2.x (MCPServer)."""


def make_server(name: str):
    try:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x

        return FastMCP(name)
    except ImportError:
        from mcp.server.mcpserver import MCPServer  # mcp 2.x

        return MCPServer(name)
