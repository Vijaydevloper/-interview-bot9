# MCP

Model Context Protocol

# What is MCP?

MCP is an open protocol standardizing how apps provide context to large language models (LLMs). It acts like a USB-C port for AI, connecting LLMs to diverse data sources and tools seamlessly.

Confidential C

# Why MCP?

Pre-built Integrations

Your LLM connects directly to many data sources and tools.

Flexibility

Easily switch between different LLM providers and vendors.

Security Best Practices

Protect your data within your own infrastructure securely.

Confidential C

# General Architecture

This diagram illustrates how MCP components interact in a connected AI environment.

Confidential C

# MCP Key Components

MCP Hosts

Apps like IDEs and AI tools that access data through MCP.

MCP Clients

Clients maintaining one-on-one connections with MCP servers.

MCP Servers

Lightweight programs exposing capabilities via MCP protocol.

Local Data Sources

Your files and databases accessed securely by MCP servers.

Remote Services

External systems accessible over the internet via APIs.

Confidential C

# Core Components

Protocol Layer

Manages message framing and request/response linking

Includes Protocol, Server, and Client classes

Confidential C

# Transport Layer

The transport layer handles the actual communication between clients and servers. MCP supports multiple transport mechanisms:

1. Stdio transport

Uses standard input/output for communication

Ideal for local processes

2. HTTP with SSE transport

Uses Server-Sent Events for server-to-client messages

HTTP POST for client-to-server messages

Confidential C

# MCP Message Types

1

Requests

Expect a response from the receiver.

2

Results

Successful responses to requests.

3

Errors

Indicate request failures.

4

Notifications

One-way messages without expected responses.

Confidential C

# MCP WORKFLOW



# Thank You

Thank you for your time and consideration. We're excited about the possibilities of empowering AI through MCP.
