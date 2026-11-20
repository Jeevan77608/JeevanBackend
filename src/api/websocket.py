"""
WebSocket handler for real-time MCP communication.

Provides bidirectional communication for:
- Tool execution with streaming responses
- Session updates
- Progress notifications
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from src.mcp_tools import ToolContext, create_all_tools
from src.models.user import SessionContext
from src.session.exceptions import SessionError, SessionNotFoundError
from src.session.manager import SessionManager

logger = logging.getLogger(__name__)


# =============================================================================
# Message Types
# =============================================================================


class MessageType(str, Enum):
    """WebSocket message types."""

    # Client -> Server
    AUTHENTICATE = "authenticate"
    EXECUTE_TOOL = "execute_tool"
    LIST_TOOLS = "list_tools"
    PING = "ping"

    # Server -> Client
    AUTHENTICATED = "authenticated"
    TOOL_RESULT = "tool_result"
    TOOLS_LIST = "tools_list"
    ERROR = "error"
    PONG = "pong"
    PROGRESS = "progress"


@dataclass
class WSMessage:
    """WebSocket message structure."""

    type: MessageType
    data: dict[str, Any] | None = None
    request_id: str | None = None

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "request_id": self.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    @classmethod
    def from_json(cls, raw: str) -> WSMessage:
        """Deserialize from JSON."""
        data = json.loads(raw)
        return cls(
            type=MessageType(data["type"]),
            data=data.get("data"),
            request_id=data.get("request_id"),
        )


# =============================================================================
# Connection Manager
# =============================================================================


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._sessions: dict[str, SessionContext] = {}

    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
    ) -> None:
        """Accept a new connection."""
        await websocket.accept()
        self._connections[connection_id] = websocket
        logger.info(f"WebSocket connected: {connection_id}")

    def disconnect(self, connection_id: str) -> None:
        """Remove a connection."""
        self._connections.pop(connection_id, None)
        self._sessions.pop(connection_id, None)
        logger.info(f"WebSocket disconnected: {connection_id}")

    def set_session(
        self,
        connection_id: str,
        session: SessionContext,
    ) -> None:
        """Associate session with connection."""
        self._sessions[connection_id] = session

    def get_session(self, connection_id: str) -> SessionContext | None:
        """Get session for connection."""
        return self._sessions.get(connection_id)

    async def send(
        self,
        connection_id: str,
        message: WSMessage,
    ) -> None:
        """Send message to a connection."""
        websocket = self._connections.get(connection_id)
        if websocket:
            await websocket.send_text(message.to_json())

    async def broadcast(self, message: WSMessage) -> None:
        """Broadcast message to all connections."""
        for websocket in self._connections.values():
            await websocket.send_text(message.to_json())


# Global connection manager
connection_manager = ConnectionManager()


# =============================================================================
# WebSocket Handler
# =============================================================================


class MCPWebSocketHandler:
    """Handles MCP protocol over WebSocket."""

    def __init__(
        self,
        session_manager: SessionManager,
        connection_manager: ConnectionManager,
    ) -> None:
        self._session_manager = session_manager
        self._connection_manager = connection_manager

    async def handle_connection(
        self,
        websocket: WebSocket,
        connection_id: str,
    ) -> None:
        """Handle a WebSocket connection. Requires authentication within 10 seconds."""
        # Check for session_id in query params for early auth
        session_id = websocket.query_params.get("session_id")
        if session_id:
            session = await self._session_manager.get_session(session_id)
            if not session:
                await websocket.close(code=4001, reason="Invalid session")
                return
            # Pre-authenticate the connection
            await self._connection_manager.connect(websocket, connection_id)
            self._connection_manager.set_session(connection_id, session)
        else:
            await self._connection_manager.connect(websocket, connection_id)

        try:
            # If not pre-authenticated, require AUTHENTICATE message within 10 seconds
            if not self._connection_manager.get_session(connection_id):
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                    msg = WSMessage.from_json(raw)
                    if msg.type != MessageType.AUTHENTICATE:
                        await self._send_error(connection_id, "First message must be authenticate")
                        await websocket.close(code=4001, reason="Authentication required")
                        self._connection_manager.disconnect(connection_id)
                        return
                    await self._handle_authenticate(connection_id, msg)
                    if not self._connection_manager.get_session(connection_id):
                        await websocket.close(code=4001, reason="Authentication failed")
                        self._connection_manager.disconnect(connection_id)
                        return
                except TimeoutError:
                    await websocket.close(code=4001, reason="Authentication timeout")
                    self._connection_manager.disconnect(connection_id)
                    return

            while True:
                raw = await websocket.receive_text()
                await self._handle_message(connection_id, raw)

        except WebSocketDisconnect:
            self._connection_manager.disconnect(connection_id)
        except json.JSONDecodeError as e:
            logger.warning("WebSocket received malformed JSON: %s", e)
            await self._send_error(connection_id, "Invalid JSON message")
            self._connection_manager.disconnect(connection_id)
        except SessionNotFoundError:
            logger.info("WebSocket session not found, disconnecting: %s", connection_id)
            await self._send_error(connection_id, "Session expired or not found")
            self._connection_manager.disconnect(connection_id)
        except SessionError as e:
            logger.warning("WebSocket session error: %s", e)
            await self._send_error(connection_id, f"Session error: {e}")
            self._connection_manager.disconnect(connection_id)
        except ValueError as e:
            logger.warning("WebSocket value error: %s", e)
            await self._send_error(connection_id, str(e))
        except Exception:
            logger.exception("Unexpected WebSocket error")
            try:
                await self._send_error(connection_id, "Internal server error")
            except Exception:
                pass
            self._connection_manager.disconnect(connection_id)

    async def _handle_message(
        self,
        connection_id: str,
        raw: str,
    ) -> None:
        """Handle incoming message."""
        try:
            message = WSMessage.from_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            await self._send_error(connection_id, f"Invalid message: {e}")
            return

        handlers = {
            MessageType.AUTHENTICATE: self._handle_authenticate,
            MessageType.EXECUTE_TOOL: self._handle_execute_tool,
            MessageType.LIST_TOOLS: self._handle_list_tools,
            MessageType.PING: self._handle_ping,
        }

        handler = handlers.get(message.type)
        if handler:
            await handler(connection_id, message)
        else:
            await self._send_error(
                connection_id,
                f"Unknown message type: {message.type}",
                message.request_id,
            )

    async def _handle_authenticate(
        self,
        connection_id: str,
        message: WSMessage,
    ) -> None:
        """Handle authentication message."""
        session_id = message.data.get("session_id") if message.data else None

        if not session_id:
            await self._send_error(
                connection_id,
                "session_id required",
                message.request_id,
            )
            return

        session = await self._session_manager.get_session(session_id)
        if not session:
            await self._send_error(
                connection_id,
                "Invalid session",
                message.request_id,
            )
            return

        self._connection_manager.set_session(connection_id, session)

        await self._connection_manager.send(
            connection_id,
            WSMessage(
                type=MessageType.AUTHENTICATED,
                data={
                    "user_id": session.user_id,
                    "project_name": session.project_name,
                },
                request_id=message.request_id,
            ),
        )

    async def _handle_execute_tool(
        self,
        connection_id: str,
        message: WSMessage,
    ) -> None:
        """Handle tool execution message."""
        session = self._connection_manager.get_session(connection_id)
        if not session:
            await self._send_error(
                connection_id,
                "Not authenticated",
                message.request_id,
            )
            return

        tool_name = message.data.get("tool_name") if message.data else None
        parameters = message.data.get("parameters", {}) if message.data else {}
        conversation_id = message.data.get("conversation_id") if message.data else None

        if not tool_name:
            await self._send_error(
                connection_id,
                "tool_name required",
                message.request_id,
            )
            return

        tools = create_all_tools(self._session_manager)
        tool = next((t for t in tools if t.definition.name == tool_name), None)

        if not tool:
            await self._send_error(
                connection_id,
                f"Tool not found: {tool_name}",
                message.request_id,
            )
            return

        context = ToolContext(
            session_id=session.session_id,
            user_id=session.user.id,
            organization_url=session.selected_org,
            project_id=session.selected_project,
            project_name=session.metadata.get("project_name"),
            access_token=(
                session.token_info.access_token if session.token_info else ""
            ),
            conversation_id=conversation_id,
        )

        try:
            result = await tool.execute(context, **parameters)

            await self._connection_manager.send(
                connection_id,
                WSMessage(
                    type=MessageType.TOOL_RESULT,
                    data={
                        "status": result.status.value,
                        "data": result.data,
                        "message": result.message,
                        "error": result.error,
                    },
                    request_id=message.request_id,
                ),
            )

        except ValueError as e:
            await self._send_error(
                connection_id, f"Invalid parameters: {e}", message.request_id
            )
        except SessionError as e:
            await self._send_error(
                connection_id, f"Session error: {e}", message.request_id
            )
        except Exception:
            logger.exception("Tool execution failed unexpectedly: %s", tool_name)
            await self._send_error(
                connection_id,
                "Tool execution failed. Please try again.",
                message.request_id,
            )

    async def _handle_list_tools(
        self,
        connection_id: str,
        message: WSMessage,
    ) -> None:
        """Handle list tools message."""
        tools = create_all_tools(self._session_manager)

        await self._connection_manager.send(
            connection_id,
            WSMessage(
                type=MessageType.TOOLS_LIST,
                data={
                    "tools": [
                        {
                            "name": t.definition.name,
                            "description": t.definition.description,
                            "category": t.definition.category,
                        }
                        for t in tools
                    ],
                },
                request_id=message.request_id,
            ),
        )

    async def _handle_ping(
        self,
        connection_id: str,
        message: WSMessage,
    ) -> None:
        """Handle ping message."""
        await self._connection_manager.send(
            connection_id,
            WSMessage(
                type=MessageType.PONG,
                request_id=message.request_id,
            ),
        )

    async def _send_error(
        self,
        connection_id: str,
        error: str,
        request_id: str | None = None,
    ) -> None:
        """Send error message."""
        await self._connection_manager.send(
            connection_id,
            WSMessage(
                type=MessageType.ERROR,
                data={"error": error},
                request_id=request_id,
            ),
        )
