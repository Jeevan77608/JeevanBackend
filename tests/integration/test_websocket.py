"""Integration tests for WebSocket MCP handler components.

Tests the building blocks of the WebSocket layer in isolation:
- MessageType enum values
- WSMessage serialization / deserialization
- ConnectionManager connect, disconnect, session management, send
- MCPWebSocketHandler initialization

These tests do **not** open a real WebSocket connection; they exercise the
component API directly using unittest mocks.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.websocket import (
    ConnectionManager,
    MCPWebSocketHandler,
    MessageType,
    WSMessage,
)


# =============================================================================
# MessageType enum
# =============================================================================


async def test_message_type_enum_values() -> None:
    """MessageType has all expected client and server message types."""
    expected_values = {
        # Client -> Server
        "authenticate",
        "execute_tool",
        "list_tools",
        "ping",
        # Server -> Client
        "authenticated",
        "tool_result",
        "tools_list",
        "error",
        "pong",
        "progress",
    }

    actual_values = {member.value for member in MessageType}
    assert actual_values == expected_values


# =============================================================================
# WSMessage serialization
# =============================================================================


async def test_ws_message_to_json_includes_type() -> None:
    """WSMessage.to_json() produces JSON containing type and timestamp keys."""
    msg = WSMessage(type=MessageType.PING, data={"hello": 1}, request_id="r1")
    raw = msg.to_json()
    parsed = json.loads(raw)

    assert parsed["type"] == "ping"
    assert "timestamp" in parsed
    assert parsed["data"] == {"hello": 1}
    assert parsed["request_id"] == "r1"


async def test_ws_message_from_json_parses_correctly() -> None:
    """WSMessage.from_json() correctly reconstructs a message from JSON."""
    payload = json.dumps({
        "type": "pong",
        "data": {"latency": 42},
        "request_id": "req-100",
    })

    msg = WSMessage.from_json(payload)

    assert msg.type == MessageType.PONG
    assert msg.data == {"latency": 42}
    assert msg.request_id == "req-100"


async def test_ws_message_from_json_invalid_type_raises() -> None:
    """WSMessage.from_json() raises ValueError for an unknown message type."""
    payload = json.dumps({
        "type": "not_a_real_type",
        "data": None,
    })

    with pytest.raises(ValueError):
        WSMessage.from_json(payload)


# =============================================================================
# ConnectionManager
# =============================================================================


async def test_connection_manager_connect_and_disconnect() -> None:
    """ConnectionManager.connect() accepts and tracks, disconnect() removes."""
    mock_ws = AsyncMock()
    manager = ConnectionManager()

    await manager.connect(mock_ws, "conn-1")
    mock_ws.accept.assert_awaited_once()

    # Connection should be tracked internally
    assert "conn-1" in manager._connections

    manager.disconnect("conn-1")
    assert "conn-1" not in manager._connections


async def test_connection_manager_set_and_get_session() -> None:
    """set_session() stores a session; get_session() retrieves it."""
    mock_ws = AsyncMock()
    manager = ConnectionManager()

    await manager.connect(mock_ws, "conn-2")

    # Before setting, get_session returns None
    assert manager.get_session("conn-2") is None

    session_stub = MagicMock()
    manager.set_session("conn-2", session_stub)
    assert manager.get_session("conn-2") is session_stub


async def test_connection_manager_send_calls_websocket() -> None:
    """send() serializes the message and calls websocket.send_text()."""
    mock_ws = AsyncMock()
    manager = ConnectionManager()

    await manager.connect(mock_ws, "conn-3")

    msg = WSMessage(type=MessageType.PONG, request_id="r-pong")
    await manager.send("conn-3", msg)

    mock_ws.send_text.assert_awaited_once()
    sent_raw = mock_ws.send_text.call_args[0][0]
    parsed = json.loads(sent_raw)
    assert parsed["type"] == "pong"
    assert parsed["request_id"] == "r-pong"


async def test_connection_manager_disconnect_clears_session() -> None:
    """disconnect() removes both the connection and its associated session."""
    mock_ws = AsyncMock()
    manager = ConnectionManager()

    await manager.connect(mock_ws, "conn-4")
    manager.set_session("conn-4", MagicMock())

    manager.disconnect("conn-4")

    assert "conn-4" not in manager._connections
    assert manager.get_session("conn-4") is None


# =============================================================================
# MCPWebSocketHandler
# =============================================================================


async def test_mcp_handler_init() -> None:
    """MCPWebSocketHandler stores references to its collaborators."""
    mock_session_manager = MagicMock()
    mock_connection_manager = MagicMock()

    handler = MCPWebSocketHandler(mock_session_manager, mock_connection_manager)

    assert handler._session_manager is mock_session_manager
    assert handler._connection_manager is mock_connection_manager
