"""
API package for Azure DevOps MCP Server.

Provides REST API and WebSocket endpoints.
"""

from src.api.routes import router
from src.api.websocket import (
    ConnectionManager,
    MCPWebSocketHandler,
    MessageType,
    WSMessage,
    connection_manager,
)

__all__ = [
    "router",
    "ConnectionManager",
    "MCPWebSocketHandler",
    "MessageType",
    "WSMessage",
    "connection_manager",
]

