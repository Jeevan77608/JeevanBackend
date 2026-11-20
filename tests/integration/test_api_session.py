"""Integration tests for session, logout, and tool-discovery endpoints."""

from __future__ import annotations

import httpx
import pytest


# Uses shared app_client fixture from tests/integration/conftest.py


# =============================================================================
# TestSessionEndpoint
# =============================================================================


class TestSessionEndpoint:
    """Tests for GET /api/v1/session when authentication is disabled."""

    async def test_get_session_auth_disabled(
        self,
        app_client: httpx.AsyncClient,
    ) -> None:
        """GET /api/v1/session returns 200 with mock session data when AUTH_ENABLED=false."""
        response = await app_client.get("/api/v1/session")
        assert response.status_code == 200

        data = response.json()
        assert "session_id" in data
        assert data["session_id"] == "dev-session-no-auth"

    async def test_get_session_fields(
        self,
        app_client: httpx.AsyncClient,
    ) -> None:
        """Session response includes session_id, user_id, user_name, and organization_url."""
        response = await app_client.get("/api/v1/session")
        assert response.status_code == 200

        data = response.json()
        assert "session_id" in data
        assert "user_id" in data
        assert "user_name" in data
        assert "organization_url" in data

        # Verify mock user values are reasonable strings
        assert isinstance(data["user_id"], str)
        assert isinstance(data["user_name"], str)


# =============================================================================
# TestLogoutEndpoint
# =============================================================================


class TestLogoutEndpoint:
    """Tests for DELETE /api/v1/session."""

    async def test_logout_returns_success_message(
        self,
        app_client: httpx.AsyncClient,
    ) -> None:
        """DELETE /api/v1/session returns a success message when auth is disabled."""
        response = await app_client.delete("/api/v1/session")

        # When AUTH_ENABLED=false the dependency returns a mock session.
        # The session_manager.destroy_session may or may not find the mock id,
        # but the endpoint itself should not raise a 500.
        assert response.status_code == 200

        data = response.json()
        assert "message" in data


# =============================================================================
# TestToolsEndpoint
# =============================================================================


class TestToolsEndpoint:
    """Tests for GET /api/v1/tools."""

    async def test_tools_list_returns_array(
        self,
        app_client: httpx.AsyncClient,
    ) -> None:
        """GET /api/v1/tools returns a list where each entry has name, description, category."""
        response = await app_client.get("/api/v1/tools")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Verify the shape of the first tool
        first_tool = data[0]
        assert "name" in first_tool
        assert "description" in first_tool
        assert "category" in first_tool

    async def test_tools_list_has_expected_tools(
        self,
        app_client: httpx.AsyncClient,
    ) -> None:
        """The tools list includes known tools: generate_user_stories and push_to_devops."""
        response = await app_client.get("/api/v1/tools")
        assert response.status_code == 200

        data = response.json()
        tool_names = {tool["name"] for tool in data}

        assert "generate_user_stories" in tool_names, (
            f"Expected 'generate_user_stories' in tool names, got: {sorted(tool_names)}"
        )
        assert "push_to_devops" in tool_names, (
            f"Expected 'push_to_devops' in tool names, got: {sorted(tool_names)}"
        )
